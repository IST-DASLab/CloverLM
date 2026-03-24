# Speedrunning GPT3: A Preliminary Report for CloverLM-4B 

Erik Schultheis<sup>1</sup>, Matin Ansaripour<sup>1</sup>, Andrei Panferov  
ISTA  
Austria

Georgios Vlassis<sup>1</sup>  
ETH Zurich  
Switzerland

Dan Alistarh  
ISTA  
Austria

<sup>1</sup> These authors contributed equally. GV wrote the initial codebase and handled data preparation; EK led the QuartetII integration and monitored the large runs, while MA led the model evaluations and performed several engineering tasks.

## Abstract

We describe a system to pretrain a 4B-parameter model called CloverLM to performance similar to the standard GPT3-175 / OPT-175B models, when measured on popular zero-shot tasks, in a highly cost-effective manner. Our approach works by combining multiple known techniques: (1) Accurate native NVFP4 training via the Quartet II algorithm \[QuartetII\]; (2) High-quality data training on the CLIMB dataset \[Climb\]; (3) Several model- and framework-specific optimizations.

While we claim no technical novelty, and our results are not highly surprising, we believe it is notable that we can reach OPT-175B-level of accuracy in pure NVFP4 in approximately 1600 B300 GPU hours, for an estimated cost between $5'600 (spot) and $10'000 (on-demand), for the main run, on the Verda cloud provider \[Verda\]. Our code is available at [https://github.com/IST-DASLab/CloverLM](https://github.com/IST-DASLab/CloverLM).

## Overview

Large language model (LLM) pretraining usually deals with maximizing parameters, tokens, and GPU hours. In this project, we investigate a narrower question: how cost-effectively can we pretrain a model that reaches similar performance, measured on standard zero-shot tasks, of GPT3-175B \[GPT3\], the model recognized as a breakthrough in language modelling, or that of its open counterpart called OPT-175 \[OPT\]?

For this, we combine known techniques: (1) a strong data mixture automatically-optimized by NVIDIA, called CLIMB/ClimbMix \[Climb\], (2) native training in NVIDIA's efficient low-precision NVFP4 format, as enabled by our existing Quartet II technique \[QuartetII\], and (3) a modest amount of training-system engineering.

The contribution of this project is empirical: we assemble an existing low-precision recipe, an existing high-quality dataset, and a simple decoder-only model, and test whether the resulting training run remains stable and competitive at nontrivial scale.

While we do not claim any technical novelty, we believe that the outcome is noteworthy. Specifically, we show that a standard dense 4B-parameter model trained for about 310B tokens on 8 x B300 GPUs reaches a compact zero-shot average that is slightly (0.6%) above OPT-175B on the most directly comparable aggregate we report, while remaining slightly below (1%) GPT-3 175B on the stricter historical aggregate. At current pricing on the Verda cloud service \[Verda\], the core eight-day run to train this model costs about $10.7k on-demand or roughly $4.6k on spot pricing. This cost ignores prior ablations, cooldown branches, and separate evaluation jobs \[Verda\], whose cost we estimate to around $1.5k. The main claim of this report is that the overall cost of reaching this quality regime can be pushed much lower than intuition might suggest.

Besides this positive result, our results also show the limitations of kernel-level low-precision speedups, which can sound larger than the end-to-end improvement one actually gets in a pretraining stack. In our experiments, the realized system-level gain over BF16 (25-50%) is substantial but clearly smaller than the raw GEMM (3-4x). This is because of the small model size, but also because the optimizer overhead, attention, communication, and data movement become bottlenecks once matrix multiplications are accelerated.

## Technical details

### Data selection

We use ClimbMix, the compact high-quality mixture released by NVIDIA together with CLIMB \[Climb\]. We tokenize ClimbMix with a [tokenizer](https://huggingface.co/gvlassis/tokenmonster/resolve/main/englishcode-32000-strict-nocapcode-v1-eot%3D14199.vocab) from the TokenMonster library \[TokenMonster\], which uses an ungreedy subword tokenization algorithm rather than standard BPE, simultaneously considering multiple alternative segmentations. The same tokenizer setup was also used in \[Beyond\].

During development, we relied on several tokenized subsets for debugging and schedule validation: a small [10M-document shard](https://drive.google.com/drive/folders/1gje5UpFHehY-huibLdOKJWSTeSHRrSmU?usp=drive_link) (about 5.5B tokens), a [100M-document shard](https://drive.google.com/drive/folders/1vyaTDPZKJdZP3eViVkvEgnkVcmzS_FDT?usp=drive_link) (around 55B tokens), and the [full 553M-document corpus](https://drive.google.com/drive/folders/104QlBZRcPAF3PQV2W1GuZIDTdV4dOIZ9?usp=sharing) (approximately 305B tokens). The first complete 4B-family validation run used the 100M-document shard for 160k steps, i.e. 83.9B sampled training tokens. The full run reported in this note used the full tokenized corpus for 590k steps, i.e. roughly 310B sampled training tokens.

The current training pipeline treats the corpus as a large concatenated token stream, with `[EOT]` document separators, and draws fixed-length windows at random. This choice simplified the implementation of distributed training and matched earlier proxy experiments, but it also means that our runs perform _random chunk sampling_ rather than strict sequential pass over a document collection.

The quality of the data mixture is a primary reason for the cost reductions reported in this project. The unusually strong behavior on certain datasets (notably ARC-E) seen later in evaluation is unlikely to be explained by the tokenizer choice or model architecture.

ClimbMix is built from Nemotron-CC \[NemotronCC\] and smollm-corpus \[SmolLM\]. The datasets are separately deduplicated, but \[Climb\] does not perform joint deduplication on their union (even though this might automatically be discovered by the CLIMB filtering). In addition, both upstream datasets do not investigate benchmark contamination. ClimbMix further elevates contamination risk, by optimizing the mixture weights to minimize aggregate benchmark scores. Thus, our results should be interpreted with caution due to potential "benchmaxxing" given the structure of ClimbMix.

### Model definition and optimizer

Our reference model family is a dense OLMo2-style \[OLMO2\] decoder-only Transformer around 4B parameters with 29 blocks, Group Query Attention (GQA) \[GQA\] with 28 KV heads and 7 Q heads, $\mathrm{ReLU}^2$ activation, and aspect ratio 4. We trained at context length 1024 in order to keep attention cost manageable, with tied embeddings. The same architecture choice was also used in \[Beyond\].

The model uses Quartet II NVFP4 linear layers for the bulk of dense compute \[QuartetII\]. In practice, the full recipe is not "full FP4": as in prior low-precision training work, some numerically sensitive components remain in higher precision, such as norm layers, as well as the LM-head. _All_ main transformer blocks use FP4 for their matrix multiplications in Q, K, V, attention out, up, and down projections. In Quartet II, the forward pass uses four-over-six \[FourOverSix\] rounding during quantization, and the backward pass combines re-quantization of forward-pass matrices, Hadamard transform, and EDEN \[EDEN\] rounding for unbiased gradient computation.

The optimizer was Adam with a peak learning rate of $3\\times 10^{-3}$, a warmup of 2000 steps, and a linear cooldown of 20000 steps. For intermediate checkpoint cooldowns, we used 5k steps, as the difference to 10k was minor for the first checkpoint, and the reduced number of steps keeps the training cost overhead of intermediate cooldowns manageable, allowing us to run cooldowns every 100k steps.

### Run progress

#### Preliminary runs.

The project reached the final configuration for the long run in two stages. First, we executed smaller 0.5B-, 1.5B-, and 4B-family runs to debug kernel integration, dataloading, and distributed evaluation. Second, we launched a complete 4B-family validation run on the 55B-token ClimbMix100m shard. That run completed 160k steps (83.9B sampled tokens) stably and finished with training loss 2.3389 and validation loss 2.2948\. This run showed that the recipe and code could handle 4B-scale pretraining run.

#### Main run.

The main run then targeted the fully tokenized ClimbMix corpus and executed 590k steps, i.e. 309.3B sampled training tokens. Early in the run, the system sustained roughly 434k tokens/s in aggregate (about 54k tokens/s/GPU); later checkpoints remained in the same general 50-54k tokens/s/GPU regime. At this throughput, the wall-clock duration is about eight days on a single 8 x B300 server. The run produced checkpoints at 200k, 300k, 400k, 500k, and 590k steps, with optional 5k-step cooldown branches from several of the intermediate checkpoints.

The overall training trajectory was surprisingly smooth, despite NVFP4 training. We encountered infrastructure failures during ablations, including CUDA/NVLink issues on one node and a late NaN event near 288.8B tokens in one continuation. But the important point is that the main recipe did not show recurring loss spikes or consistent instability. In particular, we believe the NaN to be caused by a division where a small divisor was flushed to zero in the quantization code, and it did not reappear after guarding these divisions for subnormals.

One clear result is that the end-to-end speedup is (much) smaller than kernel-level speedup. In controlled comparisons, NVFP4 delivered roughly 25%-50% higher end-to-end throughput than BF16, depending on model size and machine configuration. This is because, once the dense matrix multiplications are accelerated, the optimizer, attention, communication, logging, and data movement become bottlenecks.

### Final evaluations

Evaluation was performed offline by converting training checkpoints into a standard model format and running a standard zero-shot evaluation pipeline. Our eval suite is fairly narrow but standard, consisting of the ARC-Challenge, ARC-Easy, HellaSwag, and PIQA tasks. We chose this suite because it provides an early and relatively stable signal for models of this size, and because it is the subset for which historical GPT-3 and OPT-175B comparisons were available.

One important detail is that these baselines do not all use the same scoring convention, specifically for ARC. Since we do not have ARC values that are comparable between our GPT3 and OPT baselines, we report two ARC variants and therefore two corresponding averages. For readability, we call them the _OPT-style_ and _GPT-3-style_ aggregates.

<div class="table-wrap">
  <table class="report-table report-table-wide">
    <caption><strong>Table 1.</strong> Compact zero-shot comparison across checkpoints. For ARC we report two variants because historical baselines use different conventions.</caption>
    <thead>
      <tr>
        <th>Checkpoint</th>
        <th>ARC-C (OPT)</th>
        <th>ARC-C (GPT-3)</th>
        <th>ARC-E (OPT)</th>
        <th>ARC-E (GPT-3)</th>
        <th>HellaSwag</th>
        <th>PIQA</th>
        <th>Average (OPT Eval.)</th>
        <th>Average (GPT-3 Eval.)</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>100k</td><td>41.8</td><td>46.4</td><td>76.2</td><td>68.1</td><td>65.5</td><td>78.8</td><td>65.6</td><td>64.7</td></tr>
      <tr><td>100k + 5k cooldown</td><td>43.7</td><td>47.4</td><td>77.0</td><td>68.4</td><td>67.0</td><td>79.2</td><td>66.7</td><td>65.5</td></tr>
      <tr><td>200k</td><td>44.0</td><td>48.5</td><td>77.8</td><td>69.0</td><td>68.5</td><td>79.7</td><td>67.5</td><td>66.4</td></tr>
      <tr><td>200k + 5k cooldown</td><td>44.9</td><td>48.0</td><td>78.3</td><td>70.5</td><td>69.2</td><td>80.2</td><td>68.2</td><td>67.0</td></tr>
      <tr><td>300k</td><td>44.1</td><td>47.4</td><td>78.7</td><td>70.5</td><td>69.4</td><td>80.3</td><td>68.1</td><td>66.9</td></tr>
      <tr><td>300k + 5k cooldown</td><td>46.1</td><td>48.1</td><td>79.0</td><td>72.4</td><td>70.2</td><td>79.3</td><td>68.6</td><td>67.5</td></tr>
      <tr><td>400k</td><td>45.6</td><td>48.6</td><td>78.9</td><td>71.4</td><td>70.3</td><td>79.9</td><td>68.7</td><td>67.6</td></tr>
      <tr><td>400k + 5k cooldown</td><td>44.5</td><td>48.7</td><td>79.4</td><td>71.1</td><td>71.1</td><td>79.8</td><td>68.7</td><td>67.7</td></tr>
      <tr><td>500k</td><td>45.6</td><td>50.9</td><td>79.4</td><td>71.5</td><td>71.3</td><td>79.5</td><td>68.9</td><td>68.3</td></tr>
      <tr><td>500k + 5k cooldown</td><td>44.9</td><td>49.6</td><td>79.5</td><td>71.3</td><td>70.8</td><td>80.4</td><td>68.9</td><td>68.0</td></tr>
      <tr><td>590k</td><td><strong>46.3</strong></td><td>50.9</td><td><strong>80.0</strong></td><td><strong>72.4</strong></td><td>71.7</td><td>80.6</td><td><strong>69.6</strong></td><td>68.9</td></tr>
      <tr><td>OPT-175B</td><td>41.2</td><td>--</td><td>75.1</td><td>--</td><td>78.3</td><td><strong>81.2</strong></td><td>69.0</td><td>--</td></tr>
      <tr><td>GPT-3 175B</td><td>--</td><td><strong>51.4</strong></td><td>--</td><td>68.8</td><td><strong>78.9</strong></td><td>81.0</td><td>--</td><td><strong>70.0</strong></td></tr>
    </tbody>
  </table>
</div>

Table 1 contains the main empirical picture. On the OPT-style aggregate, the final 590k checkpoint reaches 69.6, slightly above the 69.0 reference for OPT-175B. On the GPT-3-style aggregate, the same checkpoint reaches 68.9, below the GPT-3 175B reference of 70.0\.

Evaluations were run using the EleutherAI Language Model Evaluation Harness (`lm-eval` v0.4.11) \[LMEval\] in zero-shot scenario. Training checkpoints stored in PyTorch Distributed Checkpoint (DCP) format were first converted to flat `.pt` state dicts, then transformed into a HuggingFace-compatible model. The converted model was loaded via `accelerate launch` in `bfloat16`, using the Quartet II pseudoquantization backend for inference. A custom `lm-eval` model wrapper pads all input sequences to multiples of 128 tokens (required by the Quartet II kernels).

For the ARC tasks, we use custom YAML task definitions that extend the standard `allenai/ai2_arc` dataset with the `acc_mutual_info` metric in addition to `acc` and `acc_norm`. All other tasks use the default `lm-eval` task definitions. All the tasks were evaluated using the default `lm-eval` templates.

The "OPT-style" ARC numbers use the `acc` metric, while the "GPT-3-style" ARC numbers use `acc_mutual_info` from the same tasks. In Table 1, HellaSwag and PIQA are reported with `acc_norm`. Accordingly, `Avg (OPT Eval.)` averages ARC-C `acc`, ARC-E `acc`, HellaSwag `acc_norm`, and PIQA `acc_norm`, whereas `Avg (GPT-3 Eval.)` averages ARC-C `acc_mutual_info`, ARC-E `acc_mutual_info`, HellaSwag `acc_norm`, and PIQA `acc_norm`. The OPT-175B baselines were sourced from the BigScience evaluation repository: [https://github.com/bigscience-workshop/bigscience/blob/master/evaluation/results/tr11/opt/bslmeval.json](https://github.com/bigscience-workshop/bigscience/blob/master/evaluation/results/tr11/opt/bslmeval.json).

Table 2 shows results on additional benchmarks evaluated every 100k steps from the same checkpoints. These include Wikitext bits-per-byte (BPB), LAMBADA (OpenAI variant, `lambada_openai`), NQ (exact match). While the model improves steadily on all metrics, the gap to GPT-3 175B is significantly larger on knowledge-intensive and generative tasks than on the multiple-choice suite reported in Table 1. This is expected: the model has 44x fewer parameters and was trained with a 1024-token context, both of which limit knowledge storage and long-range coherence. Note that all CloverLM results in this table use the pseudoquantization backend, whereas the final 590k row in Table 1 was evaluated with real Quartet II NVFP4 kernels.

We also evaluated MMLU \[MMLU\] at the final checkpoint. Table 3 reports the results. Few-shot MMLU accuracy reaches 41.9%, substantially above the 31.8% reported for OPT-175B ([https://github.com/LudwigStumpp/llm-leaderboard](https://github.com/LudwigStumpp/llm-leaderboard)) and approaching the 43.9% reported in \[MMLU\] for GPT-3 175B \[GPT3\]. The "continuation" variant, which scores answer choices by continuation likelihood rather than full-sequence likelihood, yields slightly lower numbers (37.8% few-shot).

<div class="table-wrap">
<table class="report-table">
<caption><strong>Table 2.</strong> Extended zero-shot evaluation results. Wiki BPB is Wikitext bits-per-byte. LAMBADA uses the OpenAI variant (<code>lambada_openai</code>). GPT-3 baselines are from [GPT3].</caption>
<thead>
<tr>
<th>Step</th>
<th>Wiki BPB (&#8595;)</th>
<th>LAMBADA acc (&#8593;)</th>
<th>NQ EM (&#8593;)</th>
</tr>
</thead>
<tbody>
<tr><td>100k</td><td>0.777</td><td>55.9</td><td>4.6</td></tr>
<tr><td>200k</td><td>0.756</td><td>57.3</td><td>5.7</td></tr>
<tr><td>300k</td><td>0.745</td><td>58.6</td><td>5.9</td></tr>
<tr><td>400k</td><td>0.739</td><td>58.8</td><td>6.9</td></tr>
<tr><td>500k</td><td>0.734</td><td>59.6</td><td>7.5</td></tr>
<tr><td>590k</td><td><strong>0.723</strong></td><td><strong>61.1</strong></td><td><strong>7.8</strong></td></tr>
<tr><td>GPT-3 175B</td><td>--</td><td><strong>76.2</strong></td><td><strong>14.6</strong></td></tr>
</tbody>
</table>
</div>

<div class="table-wrap">
<table class="report-table">
<caption><strong>Table 3.</strong> MMLU results at the final checkpoint.</caption>
<thead>
<tr>
<th>Category</th>
<th>MMLU 0-shot</th>
<th>MMLU few-shot</th>
<th>MMLU (continuation) 0-shot</th>
<th>MMLU (continuation) few-shot</th>
</tr>
</thead>
<tbody>
<tr><td>Humanities</td><td>35.4</td><td>35.7</td><td>30.1</td><td>30.2</td></tr>
<tr><td>Social Sciences</td><td>42.1</td><td>47.1</td><td>40.8</td><td>42.4</td></tr>
<tr><td>STEM</td><td>37.2</td><td>39.0</td><td>35.5</td><td>35.8</td></tr>
<tr><td>Other</td><td>45.2</td><td>49.1</td><td>45.5</td><td>46.9</td></tr>
<tr><td><strong>Overall</strong></td><td>39.4</td><td><strong>41.9</strong></td><td>37.1</td><td>37.8</td></tr>
<tr><td>OPT-175B</td><td>--</td><td>31.8</td><td>--</td><td>--</td></tr>
<tr><td>GPT-3 175B</td><td>--</td><td><strong>43.9</strong></td><td>--</td><td>--</td></tr>
</tbody>
</table>
</div>

### Cost estimation

The main run lasted about eight days on a single 8 x B300 node, corresponding to approximately 192 instance-hours or 1,536 GPU-hours. For cost accounting, the effective rate that applied when the experiment was actually run was $5.70 per GPU-hour. Equivalently, this is $45.60 per hour for the full 8-GPU node, which yields an estimated cost of $8,755.20 for the full run. Note that Verda currently lists higher B300 pricing at $6.99 per GPU-hour on-demand \[Verda\]. (See Table 4.) The true project cost is higher once one includes the earlier 83.9B-token validation run, smaller debugging runs, shape ablations, cooldown jobs, and separate evaluation machines. We estimate this cost to an additional $2k.

<div class="table-wrap">
  <table class="report-table">
    <caption><strong>Table 4.</strong> End-to-end run cost for the 309.3B-token training run.</caption>
    <thead>
      <tr>
        <th>Scenario</th>
        <th>GPU-hours</th>
        <th>Rate</th>
        <th>Cost</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>Run-time cost (standard)</td><td>1,536</td><td>$5.70 / GPU-h</td><td>$8,755.20</td></tr>
      <tr><td>Run-time cost (spot)</td><td>1,536</td><td>$3.00 / GPU-h</td><td>$4,608.00</td></tr>
    </tbody>
  </table>
</div>

## Conclusion

This report documents a "speedrun" attempt for a model of the GPT3-175B quality, finding that, if one combines a strong public data mixture and a working native NVFP4 recipe, a roughly 4B parameter model can be pretrained stably for about 309.3B tokens and reach a compact zero-shot average that is at least competitive with OPT-175B, at a core-run cost that is low by historical standards.

We caveat our findings on multiple points: We fall slightly short of GPT-3 parity, and the current recipe is clearly stronger on short-context multiple-choice tasks than on open-ended generation, suggesting somewhat biased data selection.

Even with these caveats, the results suggest that a combination of good data, native low precision, and careful engineering can compress pretraining cost significantly.

In future work, we plan to extend our study to larger model sizes, and potentially different architectures.

## Acknowledgments

The authors would like Verda Cloud for computational support, and in particular Paul Chang for his consistent, prompt and generous help throughout the project. We also thank Jen Iofinova for performing basic safety testing on our model. Andrei Panferov and Erik Schultheis are supported in part by the BilAI (Biliteral AI) Austrian Center of Excellence in Artificial Intelligence, while Georgios Vlassis is supported in part by SwissAI.

## References

* `[GPT3]` Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. _Language Models are Few-Shot Learners._ In _Advances in Neural Information Processing Systems_, 2020\.
* `[OPT]` Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. _OPT: Open Pre-trained Transformer Language Models._ _arXiv preprint arXiv:2205.01068_, 2022\.
* `[QuartetII]` Andrei Panferov, Erik Schultheis, Soroush Tabesh, and Dan Alistarh. _Quartet II: Accurate LLM Pre-Training in NVFP4 by Improved Unbiased Gradient Estimation._ _arXiv preprint arXiv:2601.22813_, 2026\.
* `[Climb]` Shizhe Diao, Yu Yang, Yonggan Fu, Xin Dong, Dan Su, Markus Kliegl, Zijia Chen, Peter Belcak, Yoshi Suhara, Hongxu Yin, Mostofa Patwary, Celine Lin, Jan Kautz, and Pavlo Molchanov. _CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training._ _arXiv preprint arXiv:2504.13161_, 2025\.
* `[SmolLM]` Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, et al. _SmolLM -- blazingly fast and remarkably powerful._ Hugging Face blog, 2024\. [https://huggingface.co/blog/smollm](https://huggingface.co/blog/smollm)
* `[EuroLLM]` Pedro Henrique Martins, Duarte M. Alves, Hippolyte Gisserot-Boukhlef, et al. _EuroLLM-9B: Technical Report._ _arXiv preprint arXiv:2506.04079_, 2025\.
* `[EuroLLMProject]` EuroLLM Project. _EuroLLM: Open Source European Large Language Model._ Project website, accessed March 2026\. [https://sites.google.com/view/eurollm/home](https://sites.google.com/view/eurollm/home)
* `[Verda]` Verda. _NVIDIA B300 SXM6 pricing page._ Accessed March 2026\. [https://verda.com/b300](https://verda.com/b300)
* `[TokenMonster]` Alasdair Forsythe. _TokenMonster: Ungreedy subword tokenizer and vocabulary trainer for Python, Go & Javascript._ GitHub repository, 2023\. [https://github.com/alasdairforsythe/tokenmonster](https://github.com/alasdairforsythe/tokenmonster)
* `[MMLU]` Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. _Measuring Massive Multitask Language Understanding._ _arXiv preprint arXiv:2009.03300_, 2021\.
* `[LMEval]` Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac'h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. _A framework for few-shot language model evaluation._ Zenodo, 2024\. [https://zenodo.org/records/10256836](https://zenodo.org/records/10256836)
* `[NemotronCC]` Dan Su, Kezhi Kong, Ying Lin, Joseph Jennings, Brandon Norick, Markus Kliegl, Mostofa Patwary, Mohammad Shoeybi, and Bryan Catanzaro. _Nemotron-cc: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset._ In _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_, pages 2459-2475, 2025\.
* `[OLMO2]` Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, Shengyi Huang, Matt Jordan, et al. _2 OLMo 2 Furious._ _arXiv preprint arXiv:2501.00656_, 2024\.
* `[FourOverSix]` Jack Cook, Junxian Guo, Guangxuan Xiao, Yujun Lin, and Song Ha. _Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling._ _arXiv preprint arXiv:2512.02010_, 2025\.
* `[EDEN]` Shay Vargaftik, Ran Ben Basat, Amit Portnoy, Gal Mendelson, Yaniv Ben Itzhak, and Michael Mitzenmacher. _EDEN: Communication-efficient and robust distributed mean estimation for federated learning._ In _International Conference on Machine Learning_, 2022\.
* `[GQA]` Joshua Ainslie, James Lee-Thorp, Michiel De Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. _GQA: Training generalized multi-query transformer models from multi-head checkpoints._ In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_, pages 4895-4901, 2023\.
* `[Beyond]` Georgios Vlassis, Saleh Ashkboos, Alexandra Volkova, Torsten Hoefler, and Dan Alistarh. _Beyond outliers: A study of optimizers under quantization._ _arXiv preprint arXiv:2509.23500_, 2025\.
