# The Careful Way to Store a Model's Memory Wastes Most of It

*The Vocabulary of an Inference Engineer, Part 2 of 4: Memory*

In 2023, the researchers who built the vLLM engine measured how the serving systems of the time used GPU memory. Between 60 and 80 percent of the memory set aside for conversations was holding nothing. The GPUs were full, and most of what filled them was empty space.

This is the second of four parts. [Part 1](PART_1_URL) was about time: how one GPU is shared between many requests. This part is about memory: where a model keeps its working notes on each conversation, why so much of that space was wasted, and what it costs to get it back. Part 3 is about reusing work that's already done, and Part 4 is about keeping that work and moving it between machines.

The series follows Banque Valmont, a bank invented for the purpose. Valmont runs its models on its own GPUs, so that its client data and its machines stay under its own control. A bank that makes that choice pays for every byte of GPU memory, which is why its inference engineers care how the memory is used.

## Before we start!

🦸🏻‍♀️ If this helps you ship better AI systems:

* 👏 Clap 50 times (yes, you can!) - Medium's algorithm favors this, increasing visibility to others who then discover the article.
* 🔔 Follow me on [Medium](https://medium.com/@han.heloir), [LinkedIn](https://www.linkedin.com/in/hanheloiryan/) and [subscribe](https://medium.com/@han.heloir/about) to get my latest article.

## What Carries Over From Part 1

**The terms below come from Part 1 and are used here without being explained again.**

A **GPU** is the kind of chip these models run on. Its memory holds the model and the working notes of every request being served.

A **token** is the unit a model reads and writes: a word, part of a word, or a punctuation mark. The **weights** are the model itself, a very large set of numbers that training has fixed. A **pass** is one run of the text through all the weights.

**Attention** is how a token takes the earlier tokens into account. Each token leaves behind a **key**, which says what it has to offer, and a **value**, which is the information it gives when chosen. Keys and values never change once computed, so the engine stores them in GPU memory and reuses them. That store is the **KV cache**.

A request is served in two phases. **Prefill** reads the whole prompt in one pass and fills the KV cache. **Decode** writes the answer one token per pass, and each pass adds one entry to the cache.

Part 1 asked who gets the GPU's time. The KV cache was in the background. Here it's the subject.

## Paged Attention: Most KV Memory Holds Nothing

**The careful way to give a request memory wastes most of that memory.**

Recall how the cache grows. Every token of a request leaves a key and a value behind. "Payment 4471 status?" writes three entries during prefill, and each word of the answer adds one more. The engine can't know the final size in advance, because it doesn't know how long the answer will be until the model stops writing.

![KV Grows With Every Token](images/13.png)

The engine still has to put that cache somewhere. Older systems wanted each request's cache in one continuous stretch of memory, because the code that computes attention was written to read one unbroken region. That forces a guess: how big a stretch to set aside?

The safe guess is the longest answer the model is allowed to give. Think of GPU memory as 12 blocks, a block being a piece of memory of fixed size. The payment request and the translation each get five blocks reserved, in case their answers run long. Each one actually uses two. Six blocks are reserved and empty. "Draft a reply" arrives needing three blocks, finds only two free, and has to wait, on a GPU where half the memory holds nothing.

![Reserving for the Longest Answer](images/14.png)

Guessing smaller doesn't solve it. Suppose each request got exactly what it needed. Requests finish at different moments, and each one leaves a hole where its cache used to be. After a while the free memory is scattered in small gaps between the requests still running. This is called **fragmentation**. Three blocks may be free in total, but not three in a row, and a request that needs one continuous stretch can't use them.

![Free Space in Pieces](images/15.png)

[PagedAttention](https://arxiv.org/abs/2309.06180), the research paper that vLLM grew out of, removes the rule that caused both problems. A request's cache no longer has to be continuous. It can be split across any blocks that happen to be free.

The engine keeps a small list for each request, called a **block table**, that records where each part is. The reply sees its cache as part one, part two, part three, in order. In physical memory those parts may sit in blocks 2, 3 and 7. When attention needs part three, the table says block 7. Operating systems have managed a computer's memory this way for decades, under the name paging.

![The Block Table](images/16.png)

Blocks are also handed out only when needed. A request starts with one block. When that block is full, the engine takes another from the list of free ones. In the diagram a block holds four tokens. The reply fills two blocks with its first eight tokens and takes a third when the ninth arrives. Nothing is reserved in advance, and the only empty space a request can hold is the unfilled end of its last block.

![Growth and Capped Waste](images/17.png)

The paper measured both sides. Earlier systems wasted 60 to 80 percent of their KV memory. With paging the waste fell under 4 percent. More requests fit on the same GPU at once, and throughput, the amount of work finished per second, rose 2 to 4 times with no increase in how long each user waited.

The cost is one extra lookup in the table each time the cache is read. It's small, and nobody has argued against paying it. vLLM and SGLang both work this way, which is why the rest of this series can talk about blocks without explaining them again.

## KV Quantization: Half the Bits, Twice the Room

**Once the empty space is gone, the next saving is in how each number is written down.**

A KV cache is a long list of numbers, and a computer stores every number as a row of bits. A bit is a single 0 or 1. The more bits a number gets, the more precisely it can be written. Each value in a KV cache normally gets 16 bits.

**Quantization** means storing numbers with fewer bits. vLLM can store the KV cache in a format that uses 8 bits per number. Half the bits means half the space.

The smaller format can't write every number the larger one can. It has fewer possible values to choose from, so each number is rounded to the nearest one it can express. In the diagram, 0.7832 becomes 0.8125.

![Half the Bits, a Little Rounding](images/18.png)

One rounding error like that is tiny. The question is what happens when the rounding applies to every value, in every layer of the model, for every token of a long conversation. Small errors can add up, and an answer can shift.

The [vLLM documentation](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/) says the loss of accuracy is typically minimal. It also recommends calibration, which means running sample prompts first so the engine can fit the smaller format to the numbers your model really produces.

"Typically minimal" is a statement about other people's workloads. For a bank, the test that counts is its own: do the compliance answers and the risk summaries stay the same with the 8 bit cache? Running that test is the engineer's work, and the documentation can't do it.

What you get in exchange is room. The same GPU memory now holds roughly twice as much KV cache. That can be spent two ways. One is longer **context**, which is the amount of text a single request can carry. The other is more requests on the GPU at the same time.

![Twice the Tokens in the Same Memory](images/19.png)

That's a gain in capacity. Whether each answer also arrives faster is a separate matter. It depends on how the engine's attention code reads the smaller format, and the engine's own documentation is the place to check before promising it to anyone.

## Decode Context Parallelism: When More GPUs Means More Copies

**Spreading a model over more GPUs can multiply a request's KV cache instead of dividing it.**

Two ideas are needed first.

A model doesn't run attention once per layer. It runs several attention computations side by side, each one free to look for a different kind of link between tokens. Each of these is called a **head**.

Some models are too large or too slow for one GPU. The standard answer is **tensor parallelism**: each layer's work is split among several GPUs, and they compute it together. The KV cache is split the same way, by head. With four GPUs, each one keeps the keys and values of its own share of the heads.

That split has a limit. To save memory, many recent models are designed to store keys and values for very few heads. If a model stores them for fewer heads than there are GPUs, there's nothing left to divide. Every GPU then keeps the whole cache.

Take a case at Valmont. The risk team submits a 100,000 token credit file, and the model runs on four GPUs. With too few heads to share out, all four GPUs hold the full 100,000 token history. The bank bought four GPUs' worth of memory and is storing the same thing four times.

![Tensor Parallelism Copies the History](images/20.png)

The [vLLM guide](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment.html) gives a real case: the DeepSeek R1 model spread over 8 GPUs this way holds 8 copies of its KV cache.

**Decode context parallelism** divides the cache along a different line. It ignores heads and splits by the position of each token in the text. Token 1 goes to the first GPU, token 2 to the second, and so on around, so each of four GPUs stores every fourth token.

During decode, each new token still has to take the whole history into account. So every GPU computes attention over its own quarter of the tokens, and the four partial results are combined into one.

![Interleaved Slices, Then Merge](images/21.png)

Each GPU now stores a quarter of the history and reads a quarter of it at every step. vLLM [reports](https://vllm.ai/blog/2026-08-07-decode-context-parallelism) 3 times the throughput of plain tensor parallelism on long tasks where a model works through many steps and its prompt keeps growing.

The price is conversation between GPUs. Every decode step now includes an exchange of partial results, and that exchange grows with the number of GPUs in the split. For short prompts the copies were small and the exchange isn't worth it. For a 100,000 token file it is.

![Less per GPU, More Talk](images/22.png)

## Three Fixes, Three Prices

**As in Part 1, none of these techniques removed a cost. Each one moved it.**

Paging ended the reserved and empty space, and added a lookup in a table at every read. Quantization doubled the room, and added a rounding that has to be tested on your own answers. Decode context parallelism ended the copies, and added an exchange between GPUs at every step.

The first of those prices is so low that everyone pays it. The other two depend on the workload. A platform full of short questions gains little from either. A platform whose analysts load very long files may not work without them.

Everything here was about where the KV cache lives and how much room it takes. Part 3 is about not computing it twice. Two analysts open the same report, and the GPU reads it twice. Part 3 shows how an engine recognizes a prompt it has already seen, and how a platform sends each request to the machine that holds its cache.

Part 4 is about keeping that cache and moving it. A cache computed on Monday is deleted at 17:00 and needed again at 9:00. The fixes are the newest work in the field, including one that vLLM shipped in October 2026.

## Credits and Further Reading

* [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), 2023. The paper behind vLLM. Source for the 60 to 80 percent waste, the drop under 4 percent, and the 2 to 4 times throughput.
* [vLLM documentation on the quantized KV cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/). Source for the 8 bit format, the "typically minimal" accuracy loss, and the advice to calibrate.
* [vLLM guide to context parallel deployment](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment.html). Source for the copies under tensor parallelism and the DeepSeek R1 example.
* [vLLM, Decode Context Parallelism](https://vllm.ai/blog/2026-08-07-decode-context-parallelism), August 2026. Source for the 3 times throughput figure.
* [Inference Engineering](https://www.baseten.co/inference-engineering/) by Philip Kiely. A free book that covers the whole field.

Banque Valmont and every request attributed to it are invented. Token counts and block sizes in the diagrams are examples unless a source is named.
