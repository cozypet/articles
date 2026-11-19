# The Cheapest Computation Is the One You Don't Repeat

*The Vocabulary of an Inference Engineer, Part 3 of 4: Reuse*

Two risk analysts at Banque Valmont open the same Q3 credit risk report on the same morning. One asks the bank's model for a summary. The other asks for the top three risks. The report is about 3,000 tokens long, a token being roughly a word, and the GPU reads all of it twice.

Nothing is broken. Each request arrived with its full prompt, and the engine did what it was asked. Most prompts in a company look like this one: a long beginning that many requests share, and a short question at the end.

This is the third of four parts. [Part 1](PART_1_URL) was about time: how one GPU is shared between many requests. [Part 2](PART_2_URL) was about memory: where a model keeps its working notes on each conversation. This part is about reuse: how an engine recognizes work it has already done, and how a platform sends each request to the machine that did it. Part 4 is about keeping that work and moving it between machines.

Banque Valmont is a bank invented for this series. It runs its models on its own GPUs, so that its data and its machines stay under its own control. Every prefill it repeats is a cost it pays itself.

## Before we start!

🦸🏻‍♀️ If this helps you ship better AI systems:

* 👏 Clap 50 times (yes, you can!) - Medium's algorithm favors this, increasing visibility to others who then discover the article.
* 🔔 Follow me on [Medium](https://medium.com/@han.heloir), [LinkedIn](https://www.linkedin.com/in/hanheloiryan/) and [subscribe](https://medium.com/@han.heloir/about) to get my latest article.

## What Carries Over From Parts 1 and 2

**The terms below come from the first two parts and are used here without being explained again.**

A **GPU** is the kind of chip these models run on. Its memory holds the model and the working notes of every request being served.

A **token** is the unit a model reads and writes: a word, part of a word, or a punctuation mark. A **pass** is one run of the text through the model.

**Attention** is how a token takes the earlier tokens into account. Each token leaves behind a **key** and a **value**. They never change once computed, so the engine stores them in GPU memory. That store is the **KV cache**, and it's the model's working notes on a conversation.

A request is served in two phases. **Prefill** reads the whole prompt in one pass and fills the KV cache. **Decode** writes the answer one token per pass.

Part 2 showed that the cache is stored in **blocks**, pieces of memory of fixed size, and that each request has a **block table** listing which blocks hold its cache.

## Prefix Caching: The Cheapest Prefill Is the One You Skip

**Most prompts in a company aren't new. Until now, each request has paid for its whole prompt anyway.**

Go back to the two analysts. Each prompt is the 3,000 token report followed by a short question. The reports are identical, and only the last line differs. Without any reuse, the GPU prefills the report twice.

The part that two prompts share from their very first token is called a **prefix**. **Prefix caching** keeps the KV cache of that shared part. The first request computes it. The second request finds it, reuses it, and prefills only its own question.

![The Same Report, Read Once](images/23.png)

Why only the beginning? Because of how attention works. A token's key and value depend on every token that came before it. The cache of the report is only valid for a prompt that starts with exactly that report. If one word near the top changes, everything after it has to be computed again.

Paging, from Part 2, makes a second saving possible. When both analysts are working at the same moment, their two block tables can point at the same physical blocks. The report is stored once in GPU memory and read by both requests.

![One Copy, Two Readers](images/24.png)

In a bank, shared prefixes are everywhere. Every assistant starts with a system prompt, the standing instructions placed before any user message. Every compliance question starts with the same policy document. Every new message in a conversation starts with the whole conversation so far.

One detail from the [vLLM design notes](https://docs.vllm.ai/en/latest/design/prefix_caching/) matters in practice. Only full blocks are cached. If a prompt ends in the middle of a block, that last partial block is always computed again.

Prefix caching removes computation, and it holds memory to do it. It's also the first technique in this series whose value depends on the cache still being there when the next request arrives. The rest of this part, and all of Part 4, follow from that.

## Prefix Matching: How an Engine Recognizes a Prompt It Has Seen

**Reuse is only correct if the beginning is identical, and the engine has to check that in almost no time.**

Thousands of prompts may be cached. A new request arrives. The engine has to find the longest beginning it has already computed, and it can't compare the new prompt with every old one. The two engines in this series solve it in two different ways.

SGLang keeps cached prompts in a **radix tree**. A tree, in computing, is a structure that starts from one point and branches. In a radix tree, prompts that begin the same way share one path from the starting point and split where they first differ. A new request follows the path as far as its tokens match. Everything up to that point is reused, and the rest is prefilled.

![Walk the Tree, Reuse the Path](images/25.png)

vLLM uses **hashes**. A hash is a short fingerprint computed from a piece of data. The same data always gives the same fingerprint, and different data almost never does. Comparing two fingerprints is much faster than comparing two long texts.

vLLM cuts the prompt into blocks and computes a hash for each one. The hash of a block is built from two things: the block's own tokens, and the hash of the block before it. A new request is checked block by block from the start, and the first block that doesn't match ends the reuse.

![A Hash That Remembers Its Past](images/26.png)

Including the previous hash is what makes this safe. Take the words "Screen this payment for sanctions". After "You are Valmont's compliance assistant", they produce one KV cache. After "You are Valmont's retail assistant", the same words produce a different one, because every key and value depends on what came before.

If the hash covered only a block's own words, the engine would treat those two blocks as the same and reuse the wrong cache. Because each hash contains the one before it, a change early in the prompt changes every hash after it, and the mistake can't happen.

![Same Words, Different Context](images/27.png)

vLLM also accepts a **cache salt**. It's an extra value mixed into the hashes of a request. Two requests with different salts never match each other's cache, even with identical prompts. On a platform shared by several business lines, that's the control that keeps one line's prompts out of another's cache.

There's a practical consequence for whoever writes the prompts. Reuse stops at the first difference, so the content that never changes has to come first, and the content that changes with every request has to come last. A date or a user name placed at the top of a prompt cancels the reuse of everything below it.

The share of requests that find a cache to reuse is called the **cache hit rate**. It's set mostly by how prompts are written, by a team the inference engineer usually doesn't manage.

## Cache Aware Routing: The Cache Was on the Other Machine

**With more than one machine, a platform that ignores where the cache is throws it away.**

One GPU machine isn't enough for a bank, so Valmont has several. The component that decides which machine serves each request is called a **router**. The simplest rule a router can follow is round robin: send each request to the next machine in turn.

Round robin is fair, and it's blind. The compliance system prompt has been computed and cached on machine A. The next compliance request is sent to machine B, because it's B's turn. Machine B has never seen that prompt, so it prefills all of it.

![The Cache Is on A, the Request Went to B](images/28.png)

A **cache aware router** keeps track of which prefixes each machine holds and sends each request to the machine that already has its beginning. The [SGLang router](https://docs.sglang.ai/advanced_features/router.html) does this by keeping its own radix tree for each machine. The tree is an estimate of what the machine has cached, built from the requests the router has sent there.

The cache can't be the only thing the router looks at. Suppose machine A holds the prefix and has twenty requests waiting. Sending one more there saves a prefill and costs a long wait in the queue. A prefill on machine B would be quicker. The SGLang router handles this by switching to the shortest queue when the load between machines is uneven.

![Cache or Queue](images/29.png)

SGLang's [version 0.4 release](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4) reported the effect of cache aware routing: up to 1.9 times the throughput, meaning the work finished per second, and a cache hit rate 3.8 times higher.

## Three Fixes, Three Prices

**As in the first two parts, none of these techniques removed a cost. Each one moved it.**

Prefix caching skips a prefill, and holds GPU memory to do it. Matching makes reuse safe, and makes the order of a prompt matter: one changed word near the top cancels everything below it. Cache aware routing keeps the reuse alive across machines, and has to give it up whenever the machine with the cache is too busy.

All three rest on one assumption: the cache still exists, and it's in the right place. Part 4 is about what happens when it isn't. An answer is cached on one group of GPUs and needed on another. A cache computed on Monday is deleted at 17:00 and needed again at 9:00. The fixes include one that vLLM shipped in October 2026, and Part 4 ends with the bill for the whole series and where to go next.

## Credits and Further Reading

* [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104). The paper that describes the radix tree approach to cache reuse.
* [vLLM prefix caching design](https://docs.vllm.ai/en/latest/design/prefix_caching/). Source for block hashing, the rule that only full blocks are cached, and cache salts.
* [SGLang router documentation](https://docs.sglang.ai/advanced_features/router.html) and the [SGLang version 0.4 release](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4). Sources for cache aware routing, the shortest queue fallback, and the 1.9 and 3.8 figures.
* [Inference Engineering](https://www.baseten.co/inference-engineering/) by Philip Kiely. A free book that covers the whole field.

Banque Valmont and every request attributed to it are invented. Token counts and block sizes in the diagrams are examples unless a source is named.
