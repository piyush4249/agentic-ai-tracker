# Agentic AI — Latency & Efficiency Digest (2025-08-26)

## Graph neural networks in TensorFlow
**ImpactScore:** 1.0  
**Source:** research.googleblog.com  
**Link:** http://blog.research.google/2024/02/graph-neural-networks-in-tensorflow.html

- </p> <a name="more"></a> <p> <a href="https://distill.pub/2021/gnn-intro/">Graph neural networks</a>, or GNNs for short, have emerged as a powerful technique to leverage both the graph’s connectivity (as in the older algorithms <a href="http://perozzi.net/projects/deepwalk/">DeepWalk</a> and <a href="https://snap.stanford.edu/node2vec/">Node2Vec</a>) and the input features on the various nodes and edges.
- <span class="byline-author">Posted by Dustin Zelle, Software Engineer, Google Research, and Arno Eigenwillig, Software Engineer, CoreML</span> <img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhcnTwrjg8cyZhVY1c-qi2ZEenIrDlkmlKlX0GsAuiKiIoxUu6i-phANh8tsCG4mUm5i-7t3zdLwuwn5DCcuQI5FKq-C3eibPnuqfoLuKFUsx-I3Ovim1Teps_JKiKZH7XqgHupnsOa2Y3peUgWcPNYG4ZIqA2_KQwxJpflo0WM6gNW8tXg5eDndiWx_dKK/s1600/TFGNN%20hero.gif" style="display: none;" /> <p> Objects and their relationships are ubiquitous in the world around us, and relationships can be as important to understanding an object as its own attributes viewed in isolation — take for example transportation networks, production networks, knowledge graphs, or social networks.
- </p> <div style="line-height: 40%;"> <br /> </div> <h2>GNNs: Making predictions for an object in context</h2> <p> For illustration, let’s look at one typical application of TF-GNN: predicting a property of a certain type of node in a graph defined by cross-referencing tables of a huge database.

---

## Mixed-input matrix multiplication performance optimizations
**ImpactScore:** 1.0  
**Source:** research.googleblog.com  
**Link:** http://blog.research.google/2024/01/mixed-input-matrix-multiplication.html

- <span class="byline-author">Posted by Manish Gupta, Staff Software Engineer, Google Research</span> <img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhEKJJf1R773hab0veY6zffF2Nf_yfV2mk8YU9yRnuBDD3ak1o0iXecWlJw2x7bL-Ez2MX1c21MXk65VMK5IsoLpJ1H6BTC6k7BvVWl_gHJpJIOG2cm3BwP4V-HCScGHYIynuskbhvu1uorQGprHGbOFmfGI7E5UWemJcZ0xSC3tC5DolBYgyBwugl6OOLr/s1180/matrixhero.png" style="display: none;" /> <p> AI-driven technologies are weaving themselves into the fabric of our daily routines, with the potential to enhance our access to knowledge and boost our overall productivity.
- For example, storing weights in the 8-bit <a href="https://en.wikipedia.org/wiki/Integer_(computer_science)">integer</a> (i.e., U8 or S8) data type reduces the memory footprint by 4× relative to <a href="https://en.wikipedia.org/wiki/Single-precision_floating-point_format">single-precision</a> (F32) and 2× relative to <a href="https://en.wikipedia.org/wiki/Half-precision_floating-point_format">half-precision</a> (F16) or <a href="https://en.wikipedia.org/wiki/Bfloat16_floating-point_format">bfloat16</a> (BF16).
- </p> <table align="center" cellpadding="0" cellspacing="0" class="tr-caption-container" style="margin-left: auto; margin-right: auto;"><tbody><tr><td style="text-align: center;"><a href="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgaLaSxuLbV_5ifXLyJsTGs0WLa23prrxrhX4IKSLZw5l3oSd2SPk5AgZtNgvUY_j-IbOyjttva-XIfkRr1cDBwCXghEz-3Q0G-6236m7_TIgTrm_K2UejYnTnhAEmZtKHq1mN9HKP0xxV8nqSxzTNHG1U0j-cVj236efpR7lSgmt082QEYNwKsGMTRiWZb/s1999/image3.png" style="margin-left: auto; margin-right: auto;"><img border="0" src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgaLaSxuLbV_5ifXLyJsTGs0WLa23prrxrhX4IKSLZw5l3oSd2SPk5AgZtNgvUY_j-IbOyjttva-XIfkRr1cDBwCXghEz-3Q0G-6236m7_TIgTrm_K2UejYnTnhAEmZtKHq1mN9HKP0xxV8nqSxzTNHG1U0j-cVj236efpR7lSgmt082QEYNwKsGMTRiWZb/s16000/image3.png" /></a></td></tr><tr><td class="tr-caption" style="text-align: center;">Memory footprint for an 175B parameter LLM model with various data types formats.</td></tr></tbody></table> <div style="line-height: 40%;"> <br /> </div> <h2>The matrix-multiply-accumulate operation</h2> <p> Modern AI hardware accelerators such as <a href="https://cloud.google.com/tpu/docs/intro-to-tpu#how_a_tpu_works">Google’s TPU</a> and <a href="https://www.nvidia.com/en-us/data-center/tensor-cores/">NVIDIA’s GPU</a> multiply matrices natively in the hardware by targeting Tensor Cores, which are specialized processing elements to accelerate matrix operations, particularly for AI workloads.

---

## Exphormer: Scaling transformers for graph-structured data
**ImpactScore:** 1.0  
**Source:** research.googleblog.com  
**Link:** http://blog.research.google/2024/01/exphormer-scaling-transformers-for.html

- <span class="byline-author">Posted by Ameya Velingker, Research Scientist, Google Research, and Balaji Venkatachalam, Software Engineer, Google</span> <img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhbovKreBr7RlKc4L36E6rLqiZBZzJSq5GLijCkomHREon5tYXd-7C2pppMXnL5Mj2d82kZGnPlarrrMzQOfRnN8kVvqDh1GnadIJ-hbaaS8VjYzCpaD-DgYor5cKx-OhTGZk9iCy5MjtwG2Q9eTyQiipDr5ViMdl2vkxfbLzWnB3wmLb8YfvVsTJ1FnOmw/s1600/EXPHORMER%2005large.gif" style="display: none;" /> <p> <a href="https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)">Graphs</a>, in which objects and their relations are represented as nodes (or vertices) and edges (or links) between pairs of nodes, are ubiquitous in computing and machine learning (ML).
- </p> <br /> <h2>Expander graphs</h2> <p> A key idea at the heart of Exphormer is the use of <a href="https://en.wikipedia.org/wiki/Expander_graph">expander graphs</a>, which are sparse yet well-connected graphs that have some useful properties — 1) the matrix representation of the graphs have similar linear-algebraic properties as a complete graph, and 2) they exhibit rapid mixing of random walks, i.e., a small number of steps in a random walk from any starting node is enough to ensure convergence to a “stable” distribution on the nodes of the graph.
- The quality of an expander graph is measured by its <em>spectral gap</em>, an algebraic property of its <a href="https://en.wikipedia.org/wiki/Adjacency_matrix">adjacency matrix</a> (a matrix representation of the graph in which rows and columns are indexed by nodes and entries indicate whether pairs of nodes are connected by an edge).

---

## Optimizing Qwen2.5-Coder Throughput with NVIDIA TensorRT-LLM Lookahead Decoding
**ImpactScore:** 0.9  
**Source:** developer.nvidia.com  
**Link:** https://developer.nvidia.com/blog/optimizing-qwen2-5-coder-throughput-with-nvidia-tensorrt-llm-lookahead-decoding/

- <img alt="" class="webfeedsFeaturedVisual wp-post-image" height="432" src="https://developer-blogs.nvidia.com/wp-content/uploads/2025/02/computer-screen-abstract-768x432.png" style="display: block; margin-bottom: 5px; clear: both;" title="computer-screen-abstract" width="768" />Large language models (LLMs) that specialize in coding have been steadily adopted into developer workflows.
- From pair programming to self-improving AI agents,...

---

## Cappy: Outperforming and boosting large multi-task language models with a small scorer
**ImpactScore:** 0.85  
**Source:** research.googleblog.com  
**Link:** http://blog.research.google/2024/03/cappy-outperforming-and-boosting-large.html

- </p> <a name="more"></a> <table align="center" cellpadding="0" cellspacing="0" class="tr-caption-container" style="margin-left: auto; margin-right: auto;"><tbody><tr><td style="text-align: center;"><a href="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhMcacnhPA68XiEskvhExF4SGFh4997UZzwvhYfXt-ReGXtzfGTamLB3LZoYSh8WWuf1dmlBnNAUecAMhrBTOMVF6vxsw3BqY8Ld5xPgSdZY_cywScxxxQ5e6uwhawA5VYDEj6VtSyOTNGZtjdLXieeFV5OLiDk3bnB-xaz4MIbvUO-7RPadk8iQDv3206V/s640/Cappy%20instruction-following.gif" style="margin-left: auto; margin-right: auto;"><img border="0" src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhMcacnhPA68XiEskvhExF4SGFh4997UZzwvhYfXt-ReGXtzfGTamLB3LZoYSh8WWuf1dmlBnNAUecAMhrBTOMVF6vxsw3BqY8Ld5xPgSdZY_cywScxxxQ5e6uwhawA5VYDEj6VtSyOTNGZtjdLXieeFV5OLiDk3bnB-xaz4MIbvUO-7RPadk8iQDv3206V/s16000/Cappy%20instruction-following.gif" /></a></td></tr><tr><td class="tr-caption" style="text-align: center;">The demonstration of the instruction-following pre-training of multi-task LLMs, e.g., FLAN.
- <span class="byline-author">Posted by Yun Zhu and Lijuan Liu, Software Engineers, Google Research</span> <img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiFNlqVAnwoYdZ97LvC4-ipR6FeOc4o9udsTUtNBBWl5Y4XHclcrz3kTCibizteSBc_xsVLh-pyRiCCNfIzTDHEs7VsJcUMCk0EjUxzvKITKCncdx1y7u9JXGkXM6TyoZY5RhUt2l_up-Us0yIV-0-EUvHsjOlFNSSNgNHlpwK1PAliqcj4gSoLsYXhIi18/s320/Cappy%20hero.jpg" style="display: none;" /> <p> Large language model (LLM) advancements have led to a new paradigm that unifies various natural language processing (NLP) tasks within an instruction-following framework.
- Pre-training tasks under this paradigm improves the performance for unseen tasks.</td></tr></tbody></table> <p> Due to the complexity of understanding and solving various tasks solely using instructions, the size of multi-task LLMs typically spans from several billion parameters to hundreds of billions (e.g., <a href="https://arxiv.org/abs/2210.11416">FLAN-11B</a>, <a href="https://arxiv.org/abs/2110.08207">T0-11B</a> and <a href="https://arxiv.org/abs/2212.12017">OPT-IML-175B</a>).

---

## Build Enterprise AI Agents with Advanced Open NVIDIA Llama Nemotron Reasoning Models
**ImpactScore:** 0.725  
**Source:** developer.nvidia.com  
**Link:** https://developer.nvidia.com/blog/build-enterprise-ai-agents-with-advanced-open-nvidia-llama-nemotron-reasoning-models/

- <img alt="" class="webfeedsFeaturedVisual wp-post-image" height="432" src="https://developer-blogs.nvidia.com/wp-content/uploads/2025/03/cloud-icon-inside-cube-768x432.png" style="display: block; margin-bottom: 5px; clear: both;" title="cloud-icon-inside-cube" width="768" />This updated post was originally published on March 18, 2025.
- Organizations are embracing AI agents to enhance productivity and streamline operations.
- To...

---
