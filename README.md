# paged-attention

ECE 759 final project. rebuilding PagedAttention (the vllm memory trick) from scratch in c++/cuda

## current idea

- llm writes 1 token at a time. attention looks back at every past token's k, v → kept around in the "kv cache"
- naive way: reserve max length (like 2048 tokens) per request upfront, one contiguous chunk. most of it sits empty → fewer users fit on the gpu
- paged way: chop memory into fixed blocks (~16 tokens each), hand them out only as a request grows. per-request block table = list of which blocks it owns. basically os virtual memory
- token t lives at `block_table[t / B]`, slot `t % B`
- waste ≤ B-1 tokens per request instead of max_len - len

## what we're NOT doing

- not implementing a full decoder in c++. attention is the only step that looks at past tokens → cut it out, test it alone w/ random q/k/v

## pieces

- cpu attention → also the correctness checker for everything else
- gpu attention → the "before" version
- block manager (cpu, bookkeeping only: unused blocks + block tables + lengths)
- gpu memory pool, one big alloc at startup
- write kernel → stores newest token's k, v into its slot
- paged attention kernel
- create fake requests to simulate realistic load, collect data, eval, etc.
- visualizer (browser) just reads the log

## checklist

- [ ] 1. cpu attention for 1 token (subtract max before exp!) + 2 sanity tests
- [ ] 2. gpu attention, naive kv
- [ ] 3. block manager + randomized tests
- [ ] 4. paged attention kernel, matches cpu ← proposal core done here
- [ ] stretch: faster kernel (close to parity with vllm?)
- [ ] 6. benchmarks: paged vs contiguous, block sizes 8–128, ms + gb/s
- [ ] stretch: side-by-side memory map visualizer (browser)
- [ ] stretch: real small model (qwen2.5-0.5b / tinyllama?) in pytorch, calls our kernel via torch extension
- [ ] stretch: split long requests across gpu blocks, share blocks for a common prompt, fp16

## demo

- lvl 1: paged attention performance improvement charts, benchmark results, etc.
- lvl 2: live side-by-side memory map, reserve-max vs paged. users served, % wasted, waiting line
- lvl 3: same viz, but real chats typing out from a real model (need to integrate our attention code with pytorch)