# A Cache Is Only Worth What It Costs to Keep and Move

*The Vocabulary of an Inference Engineer, Part 4 of 4: Transfer and Offload*

On October 12, 2026, the vLLM engine shipped a feature whose release notes contain an unusual number: 64. When two machines could share a piece of work, and the piece is smaller than 64 tokens, the engine doesn't share it. It does the work again, because fetching that little costs more than repeating it.

That number is a price list in one line. Computing something, storing it and moving it each have a cost, and an inference platform is always choosing the cheapest of the three.

This is the last of four parts. [Part 1](PART_1_URL) was about time: how one GPU is shared between many requests. [Part 2](PART_2_URL) was about memory: where a model keeps its working notes. [Part 3](PART_3_URL) was about reuse: how an engine recognizes work it has already done. This part is about keeping that work and moving it, between two groups of GPUs and down to cheaper memory. It ends with the bill for the whole series.

The series follows Banque Valmont, a bank invented for the purpose, which runs its models on its own GPUs so that its data and its machines stay under its own control.

## Before we start!

🦸🏻‍♀️ If this helps you ship better AI systems:

* 👏 Clap 50 times (yes, you can!) - Medium's algorithm favors this, increasing visibility to others who then discover the article.
* 🔔 Follow me on [Medium](https://medium.com/@han.heloir), [LinkedIn](https://www.linkedin.com/in/hanheloiryan/) and [subscribe](https://medium.com/@han.heloir/about) to get my latest article.

## What Carries Over From Parts 1 to 3

**The terms below come from the first three parts and are used here without being explained again.**

A **GPU** is the kind of chip these models run on. Its memory holds the model and the working notes of every request being served.

A **token** is the unit a model reads and writes: a word, part of a word, or a punctuation mark. A **pass** is one run of the text through the model.

**Attention** is how a token takes the earlier tokens into account. Each token leaves behind a **key** and a **value**. They never change once computed, so the engine stores them in GPU memory. That store is the **KV cache**, and it's the model's working notes on a conversation.

A request is served in two phases. **Prefill** reads the whole prompt in one pass and fills the KV cache. **Decode** writes the answer one token per pass.

Part 1 ended with **prefill and decode separation**: one group of GPUs does only prefill, another does only decode, and the KV cache is sent from the first to the second over the network.

Part 2 showed that the cache is stored in **blocks**, pieces of memory of fixed size, and that each request has a **block table** listing which blocks hold its cache.

Part 3 added **prefix caching**. When two prompts start with the same text, the engine keeps the KV cache of that shared beginning and skips its prefill the second time. Part 3 also introduced the **router**, the component that decides which machine serves each request.

## Two Way KV Transfer: Turn Two Pays for Turn One

**Separating prefill and decode has a cost that only shows up in conversations.**

A **turn** is one message from the user and the model's answer to it. A model keeps nothing in mind between turns. At turn two, the whole conversation so far is sent again as the prompt: the first message, the first answer, and the new message.

A relationship manager at Valmont asks why a client was charged 15 euros. With separation, the prefill GPUs read the question and the decode GPUs write the answer. The KV cache of that answer was built on the decode GPUs, one token at a time, and it exists only there.

Turn two is "Draft the refund letter." The prompt now contains the first answer. It goes to the prefill GPUs, as every prompt does. They never computed that answer and hold no cache for it. Without help, they compute it from scratch, while the decode GPUs next door already have it.

![The Answer Lives on the Wrong Side](images/30.png)

In Part 1, the cache moved one way: from prefill to decode. In October 2026, vLLM [added the other direction](https://vllm.ai/blog/2026-10-12-bidirectional-kvxfer-multiturn-agentic-workload). At turn two, the prefill side first fetches the existing cache from the decode side. It computes only the new message. Then it sends that new part back, so decode can carry on.

![Pull, Compute, Send](images/31.png)

The release reports a time to first token, the wait before the first word appears, up to 3.3 times shorter on long conversations.

It also sets the threshold this part opened with. If fewer than 64 tokens could be reused, the prefill side doesn't fetch anything and computes them itself. Asking another machine for a cache takes a fixed amount of time, and for a very small cache that time is longer than the computation.

That threshold states the rule behind most of this series. Moving a cache and computing it again are two prices for the same result, and the engine pays the lower one.

## KV Offload: Move the Cache, Don't Delete It

**Every form of reuse in Part 3 assumed the cache was still there.**

GPU memory is limited, and new requests keep arriving. When the engine needs room, it deletes cached data. That's called **eviction**. The first thing to go is cache that no running request is using.

Take the Q3 credit risk report, 3,000 tokens long, that Valmont's risk analysts query during the day. By Monday evening, its cache is exactly that. Nobody has asked about the report for hours, so the engine evicts it. On Tuesday at 9:00, an analyst asks to compare the report with Q2. The cache is gone, and the 3,000 token prefill runs again.

![Evicted Overnight](images/32.png)

**KV offload** moves the blocks instead of deleting them. A GPU machine has other places to keep data. Its CPU memory is the ordinary memory every server has. It's much larger than GPU memory and much cheaper. Below that is the disk, larger and cheaper again, and slower.

With offload, a cache that leaves the GPU goes down to CPU memory or to disk. When a later request matches it, the blocks are loaded back onto the GPU, and the prefill is skipped.

![Move It Down, Not Out](images/33.png)

vLLM can offload to CPU memory by itself. [LMCache](https://docs.vllm.ai/en/latest/examples/disaggregated/lmcache/), a companion project, adds disk and storage shared between machines. Shared storage also eases the routing problem from Part 3, where a request sent to a machine without the cache paid for a full prefill. A cache written by machine A can now be loaded by machine B, so the "wrong" machine costs a load instead.

![Any Machine Can Load It](images/34.png)

Offload is worth it for as long as loading a cache is faster than computing it again. For a 3,000 token report, it is. For a short prompt it may not be, which is the 64 token rule seen from another side.

## HiSparse: Keep the History Off the GPU

**Offload applies to requests that are idle. A few newer models allow it for a request that's still running.**

This last technique is the most specialized in the series. It applies to one family of models, and it's here to show where the field is heading.

In standard attention, every new token looks at the entire history. So the whole KV cache of a running request has to sit on the GPU, ready to be read at every step.

**Sparse attention** reads only part of the history. Some models, such as DeepSeek V3.2, are built this way from the start. They contain a small component called an **indexer**. At each step, the indexer gives every past token a quick score, and full attention then runs only on the few tokens that scored highest.

![Attention That Reads a Few](images/35.png)

If most of the cache isn't read at a given step, most of it doesn't need to be on the GPU. SGLang's [HiSparse](https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/) uses that. It keeps the full KV cache in CPU memory and only a small working buffer on the GPU. The indexer's own keys stay on the GPU, because the indexer reads all of them at every step.

![Full History on CPU, Working Set on GPU](images/36.png)

At each step, the indexer picks its tokens. Those already in the GPU buffer are used directly. Those that aren't are copied up from CPU memory. Everything the indexer didn't pick stays where it is.

![Fetch Only What's Missing](images/37.png)

The limits are specific, and the [SGLang guide](https://docs.sglang.io/docs/advanced_features/hisparse_guide) states them. HiSparse runs on the decode side only, and it requires prefill and decode separation. It supports a named list of sparse attention models and no others. It adds overhead when few requests are running at once. It's built for many long conversations in parallel, and it isn't something to switch on by default.

## The Bill Never Disappears

**Across all four parts, no technique removed a cost. Each one moved it.**

Continuous batching filled idle seats and let heavy prefills into light batches. Separation ended that interference and added a transfer. Paging ended the empty space and added a lookup. KV quantization halved the memory and added a risk to accuracy. Prefix caching removed computation and took memory. Offload freed memory and added loading time.

![Every Fix Moves the Bill](images/38.png)

This is why another company's configuration rarely transfers. A bank whose assistants share one long compliance prompt gains most from prefix caching and cache aware routing. A bank whose analysts submit 100,000 token credit files runs out of memory before anything else matters.

It's also why the role exists. The papers are public and the engines are free. What can't be downloaded is the knowledge of which trade a given workload needs.

The starting point is the symptom. Measure where the platform is slow, then choose the technique that gives back the resource you're short of.

![Start From Where It's Slow](images/39.png)

A decode step still fetches everything to write one token. What changed since 2022 is how many requests share that fetch, how much there is to fetch, and how often the same work is done twice. Each of those changes has a price, and it's usually in a column nobody was watching.

## A Vocabulary Is Not the Job

**These four parts covered how an engine schedules requests and manages the KV cache. The role asks for more.**

Three subjects are missing. Speculative decoding has a small model guess several tokens ahead and the large model check them in one pass. Weight quantization stores the model itself in fewer bits, where Part 2 only shrank the cache. Kernel work, such as FlashAttention, rewrites the attention computation so that less data moves in and out of GPU memory.

The next step is free. Philip Kiely's [Inference Engineering](https://www.baseten.co/inference-engineering/) covers the whole field, from GPUs to production, and can be read online. It assumes you know what a transformer is and what latency and throughput mean. This series was written to get you there.

After that, the engines are the textbook. vLLM and SGLang are open source, and the [vLLM quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/) serves a model on a single GPU. Their release notes will read differently now. Each one describes a trade, and you can name both sides.

The role takes the model as given and is measured on what each GPU and each dollar produces. That starts with being able to read the bill.

## Credits and Further Reading

* [vLLM, two way KV transfer](https://vllm.ai/blog/2026-10-12-bidirectional-kvxfer-multiturn-agentic-workload), October 2026. Source for the 3.3 times figure and the 64 token threshold.
* [LMCache with vLLM](https://docs.vllm.ai/en/latest/examples/disaggregated/lmcache/). Source for offload to disk and shared storage.
* [SGLang HiSparse](https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/) and the [HiSparse guide](https://docs.sglang.io/docs/advanced_features/hisparse_guide). Sources for the design and its limits.
* [Inference Engineering](https://www.baseten.co/inference-engineering/) by Philip Kiely. The free book recommended in the last section.

Banque Valmont and every request attributed to it are invented. Token counts and block sizes in the diagrams are examples unless a source is named.
