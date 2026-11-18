---
layout: project_page
permalink: /

# You can declare accent colors here
# accent: '#D21111'
# accent: darkorange

title: "Progressive Memory Transformer: Memory-Aware Attention for Time-Series"
authors:
    - name: Tord Sture Stangeland
      link: https://www.mn.uio.no/english/?vrtx=person-view&uid=tordss
      affiliation: 1, 2
    - name: Andreas Köhler
      link: https://en.uit.no/ansatte/andreas.kohler
      affiliation: 2, 3
    - name: Steffen Mæland
      link: https://www.hvl.no/en/employee/?user=Steffen.Meland
      affiliation: 2, 4
    - name: Adín Ramírez Rivera
      link: https://www.mn.uio.no/ifi/english/people/aca/adinr/
      affiliation: 1
affiliations:
    - name: Department of Informatics, University of Oslo
      link: https://www.mn.uio.no/ifi/english/
    - name: NORSAR
      link: https://www.norsar.no/
    - name: Department of Geosciences, UiT The Arctic University of Norway
      link: https://en.uit.no/
    - name: Western Norway University of Applied Sciences
      link: https://www.hvl.no/en/
paper: https://openreview.net/forum?id=jyIfuVt2UU
# video: https://www.youtube.com/@UniOslo
code: "{{ site.github.repository_url }}" # in case you want to use the same repo where the gh-pages is (the most common setup)
# code: https://github.com/dsb-ifi # in case you want to hard-code the repo
# data: https://huggingface.co/docs/

abstract: >-
    Time-series carry structure simultaneously at multiple scales (fine-grained variation, mid-range motifs, and global properties) and downstream tasks operate at correspondingly different scales.
    Most existing self-supervised learning approaches supervise representations globally via instance-level contrastive losses and limited temporal neighborhood supervision, but do not explicitly exploit the structural hierarchy.
    We propose a learning framework that explicitly enforces a structural hierarchy across three scales independently: a local objective for token alignment, a mid-range objective for window-level motifs, and a global objective for sequence-level agreement.
    Realizing this framework requires the backbone to expose a representation at each scale; we introduce <strong>Progressive Memory Transformer</strong> (PMT), which augments a transformer with writable, window-aligned memory that exposes the mid-range scale alongside the token and sequence-level representations conventional transformers already provide.
    Across eight UCR/UEA/UCI classification benchmarks, a cue-retention probe, and forecasting benchmarks, PMT learns representations that probe well at the global, mid-range, and local scales—strong low-label classification (1–5% labels), competitive forecasting performance across multiple horizons, and quantitative and qualitative evidence that memory states capture mid-range motifs.
---


## The problem
Data that unfolds over time, like heartbeats, factory sensor readings, or the motion signals from a phone in your pocket, has patterns at several scales at once. A walking signal has tiny wiggles from each footstep, repeating strides lasting a second or two, and an overall character that says "this person is climbing stairs." Different tasks need different scales: forecasting the next few seconds depends on the fine detail, spotting a recurring motif depends on the middle, and classifying a whole recording depends on the big picture.

Most methods that learn from this kind of data without human labels train only the big picture, asking whether two whole recordings look alike. The middle scale, where much of the meaningful structure lives, is learned by accident or pooled away. Even models that carry context forward in time, such as recurrent networks, treat what they carry as internal plumbing. Nothing in their training checks what it holds.

## Our approach
We train the model at all three scales, with one objective per scale. Each objective compares two randomly altered views of the same signal:

- **Local (HGCL).** Each token, a short patch of the signal, should match the token at the same position in the other view. Nearby tokens are similar by nature, so they are weighted by distance instead of being treated as mismatches.
- **Mid-range (PCL).** Each memory state should match the memory state of the same window in the other view.
- **Global (ICL).** The summary of the whole sequence should match across the two views.

The mid-range objective needs something to train, and a standard transformer has nothing at that scale. The Progressive Memory Transformer (PMT) adds it: a small, writable memory, aligned with windows of the signal. PMT cuts the token sequence into overlapping windows. Each window reads the memory left by the previous window, processes it together with its own tokens, and writes a new memory. Stacked blocks pass memory forward in time and up in depth, so the newest memory has seen the whole history, and learned gates let the model overwrite memory that has gone stale. Because every memory state belongs to a known window, the training can inspect and shape what it stores.

![PMT overview]({{ site.image_base_path | append: "pmt-teaser.png" | absolute_url }})

{: .figure-caption}
**Figure 1: Three scales, one objective each.** PMT turns a signal into tokens (low), window-aligned memory states (mid), and a sequence summary (high). The shaded regions show how much of the signal each level sees. Exposing the middle level (gray) lets training supervise it directly, where standard methods train only the final readout.

<video controls muted playsinline preload="metadata" poster="{{ site.image_base_path | append: "pmt-method-poster.png" | absolute_url }}" style="width: 100%;">
  <source src="{{ "/assets/videos/pmt_method_720p.mp4" | absolute_url }}" type="video/mp4">
</video>

{: .figure-caption}
**Video:** the method step by step, from the structure of time series at several scales to the three training objectives.

## Results
We train PMT without labels, freeze it, and test each scale with a task that depends on it. The comparisons are with established self-supervised methods for time series, including TS2Vec and TS-TCC trained with SoftCLT.

### Global: classification with few labels
On eight standard benchmarks from six domains, among them human activity, brain recordings, engine noise and starlight, a simple classifier reads the frozen features after seeing labels for only 1% or 5% of the examples. PMT has the highest average accuracy, 85.0% against 83.5% and 83.3% for TS-TCC and TS2Vec, and is best or tied for best in 13 of 16 settings. The largest gains are on StarLightCurves, the longest recordings in the suite: 6.6 points at 1% labels and 4.6 points at 5%.

### Local: forecasting
The token features also predict what comes next. On electricity use and electrical transformer data, with forecasts from 24 to 720 steps ahead, PMT makes smaller errors than TS2Vec with SoftCLT in 31 of 40 comparisons, including every forecast length on electricity use. The largest margin is on single-variable transformer data, where the average squared error falls from 0.155 to 0.114.

![PMT results]({{ site.image_base_path | append: "pmt-results.png" | absolute_url }}){: style="display: block; margin: 0 auto; max-width: 560px; width: 100%;"}

{: .figure-caption}
**Figure 2: Classification and forecasting.** Top: accuracy (%) of a classifier on frozen features with 1% and 5% of the labels. Bottom: average forecasting error (MSE) over five forecast lengths. Bold marks the best method in each setting.

### Mid-range: does the memory remember?
Good accuracy does not show that a model uses its memory, since a task may be solvable from local detail alone. We designed a test that isolates it. We hide a short synthetic burst in an engine-noise signal, at varying distances from its end, and ask whether the model's state at the very end still shows that the burst was there. The model was never trained on such bursts, so it can only succeed by carrying the evidence forward.

PMT detects the burst almost perfectly: 0.995, on a scale where 0.5 is guessing and 1 is perfect. It is ahead of two other designs that pass state forward, xLSTM (0.986) and Transformer-XL (0.967), and far ahead of TS2Vec, which pools the middle scale away (0.639). The lead comes from training the memory directly: without the mid-range objective, PMT scores 0.962.

<div style="max-width: 640px; margin: 1.5rem auto; font-size: 0.95rem;">
<div style="display: flex; align-items: center; margin: 0.35rem 0;"><span style="width: 11rem; text-align: right; padding-right: 0.75rem;"><strong>PMT</strong></span><span style="flex: 1; background: #eef1f4; border-radius: 4px;"><span style="display: block; width: 99.0%; background: #c2185b; border-radius: 4px; height: 1.25rem;"></span></span><span style="width: 3.5rem; padding-left: 0.6rem;"><strong>0.995</strong></span></div>
<div style="display: flex; align-items: center; margin: 0.35rem 0;"><span style="width: 11rem; text-align: right; padding-right: 0.75rem;">xLSTM</span><span style="flex: 1; background: #eef1f4; border-radius: 4px;"><span style="display: block; width: 97.2%; background: #90a4ae; border-radius: 4px; height: 1.25rem;"></span></span><span style="width: 3.5rem; padding-left: 0.6rem;">0.986</span></div>
<div style="display: flex; align-items: center; margin: 0.35rem 0;"><span style="width: 11rem; text-align: right; padding-right: 0.75rem;">Transformer-XL</span><span style="flex: 1; background: #eef1f4; border-radius: 4px;"><span style="display: block; width: 93.4%; background: #90a4ae; border-radius: 4px; height: 1.25rem;"></span></span><span style="width: 3.5rem; padding-left: 0.6rem;">0.967</span></div>
<div style="display: flex; align-items: center; margin: 0.35rem 0;"><span style="width: 11rem; text-align: right; padding-right: 0.75rem;">PMT, memory untrained</span><span style="flex: 1; background: #eef1f4; border-radius: 4px;"><span style="display: block; width: 92.4%; background: #e3a3bd; border-radius: 4px; height: 1.25rem;"></span></span><span style="width: 3.5rem; padding-left: 0.6rem;">0.962</span></div>
<div style="display: flex; align-items: center; margin: 0.35rem 0;"><span style="width: 11rem; text-align: right; padding-right: 0.75rem;">TS2Vec</span><span style="flex: 1; background: #eef1f4; border-radius: 4px;"><span style="display: block; width: 27.8%; background: #90a4ae; border-radius: 4px; height: 1.25rem;"></span></span><span style="width: 3.5rem; padding-left: 0.6rem;">0.639</span></div>
</div>

{: .figure-caption}
**Figure 3: Detecting a hidden burst from the end of the signal** (AUC, average of five training runs). Bars start at 0.5, the score of guessing. "Memory untrained" is PMT trained without the mid-range objective.

The memory also organizes itself by content. We stitched together three recordings of a person walking upstairs, downstairs, and upstairs again, and compared how the model describes each moment of the result.

![How PMT sees activities]({{ site.image_base_path | append: "pmt-hybrid-result.png" | absolute_url }})

{: .figure-caption}
**Figure 4: How PMT sees activities.** The top-left panel shows a smartphone motion recording stitched together from three recordings: walking upstairs, walking downstairs, and walking upstairs again. Each square is a heat map comparing the model's description of every moment of the stitched signal with every other, with brighter colors meaning more similar. The middle row compares the input tokens and the bottom row compares the memory states. The dashed lines mark where the pieces were joined. In the memory states, the two upstairs pieces light up together while the downstairs piece stays dark, a sharper split than in the input tokens, even though the model was never told which activity was which. The other columns compare the stitched signal with untouched recordings of each activity.

### What the memory adds
To separate the memory from the rest of the method, we kept the tokenizer, the training and the other two objectives the same, and swapped PMT's memory for the state of an xLSTM or the cache of a Transformer-XL. Neither exposes a memory that the mid-range objective can train. Over seven benchmarks, PMT averages 83.6% accuracy against 83.0% for both, and it stays ahead (83.4%) even with its own mid-range objective switched off.

The three objectives also work together: in our ablations, training with only one of them costs up to 4.4 points of accuracy. Weighting all three equally lands within 0.4 points of tuned weights, so equal weights are a good default.

## Use it
The [code]({{ site.github.repository_url }}) trains PMT from scratch with the recipe of every run in the paper, and on your own data with a two-line configuration.

```bash
python scripts/train.py --dataset UCR_PhalangesOutlinesCorrect
```

A trained encoder takes raw series of shape `[batch, channels, time]` and returns the representation at every scale.

```python
import pmt

encoder = pmt.load_encoder('runs/UCR_PhalangesOutlinesCorrect_s42/encoder.pt')
features = encoder(x)  # features.cls, features.tokens, features.pma_states
```

[Paper](https://openreview.net/forum?id=jyIfuVt2UU) · [Code]({{ site.github.repository_url }}) · [Poster]({{ site.github.repository_url }}/releases/download/v1.0.0/pmt_poster.pdf) · [Video]({{ site.github.repository_url }}/releases/download/v1.0.0/pmt_method_720p.mp4)

## Citation
{% raw %}
```
@inproceedings{
stangeland2026progressive,
title={Progressive Memory Transformer: Memory-Aware Attention for Time-Series},
author={Tord Sture Stangeland and Andreas K{\"o}hler and Steffen Maeland and Ad{\'\i}n Ram{\'\i}rez Rivera},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=jyIfuVt2UU}
}
```
{% endraw %}
