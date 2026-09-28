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
    We propose a learning framework that explicitly enforces a structural hierarchy across three scales independently: a local objective for token continuity, a mid-range objective for window-level motifs, and a global objective for sequence-level agreement.
    Realizing this framework requires the backbone to expose a representation at each scale; we introduce <strong>Progressive Memory Transformer</strong> (PMT), which augments a transformer with writable, window-aligned memory that exposes the mid-range scale alongside the token and sequence-level representations conventional transformers already provide.
    Across seven UCR/UEA/UCI classification benchmarks, a cue-retention probe, and forecasting benchmarks, PMT learns representations that probe well at the global, mid-range, and local scales—strong low-label classification (1–5% labels), competitive forecasting performance across multiple horizons, and quantitative and qualitative evidence that memory states capture mid-range motifs.
---


## The problem
Data that unfolds over time, like heartbeats, factory sensor readings, or the motion signals from a phone in your pocket, has patterns at several scales at once. A walking signal has tiny wiggles from each footstep, repeating strides lasting a second or two, and an overall character that says "this person is climbing stairs." Most methods that learn from this kind of data without human labels focus on the big picture, asking whether two whole recordings look alike. That leaves the middle scale, the recurring motifs where much of the meaningful structure lives, to be learned only by accident or squeezed away.

## Our approach
We teach the model at all three scales separately. To make this possible, we propose the Progressive Memory Transformer (PMT), a transformer that keeps a small, writable memory as it reads a signal window by window. The memory gives the model a dedicated place to store mid-range patterns, and the training can check and shape what goes into it directly.

![PMT overview]({{ site.image_base_path | append: "pmt-teaser.png" | absolute_url }})

{: .figure-caption}
**Figure 1:** PMT reads a signal window by window and learns representations at three scales (low, mid, and high). Each level carries a memory state forward in time, and the final states at the mid and high levels form the readout.

## Results
We tested PMT on seven standard classification benchmarks, where the model had to recognize categories such as human activities, engine faults, or electrical devices after seeing labels for only 1–5% of the examples. PMT had the highest average accuracy and won 11 of 14 settings. On the human activity recognition dataset, PMT trained with 1% of the labels beat competing methods that got 5%.

We also designed a new test of memory. We hid a short synthetic blip early in a signal and asked whether the model's summary at the very end still "remembered" it, even though the model had never been trained to look for such blips. PMT detected the hidden blip almost perfectly (0.98 on a scale where 0.5 is guessing). It clearly beat other memory-based designs, and it far outperformed a popular method that pools information away, which scored 0.65.

For forecasting electricity use and electrical transformer data, PMT made smaller errors than a comparable method, especially when predicting far into the future. Longer forecasts are where a memory that carries context across windows should help most.

![How the PMT sees activities]({{ site.image_base_path | append: "pmt-hybrid-result.png" | absolute_url }})

{: .figure-caption}
**Figure 2: How the PMT sees activities.** The top-left panel shows a smartphone motion recording stitched together from three separate recordings of a person walking downstairs, walking on flat ground, and walking upstairs. To a human eye, these three activities produce very similar-looking signals. Each square below is a heat map comparing how the model describes each moment of the stitched signal, with brighter colors meaning more similar descriptions. The middle row compares the model's moment-to-moment features, and the bottom row compares its memory within the model. The dashed lines mark where the pieces were joined. In the highlighted left column, where the stitched signal is compared with itself, the similarity stays inside each piece and does not spill across the joins. Both the moment-to-moment features and the memory keep the three kinds of walking apart, even though the model was never told which activity was which. The other columns compare the stitched signal with untouched recordings of each of the three activities.

## Citation
{% raw %}
```
@inproceedings{stangeland2026,
  title     = {Progressive Memory Transformer: Memory-Aware Attention for Time-Series},
  author    = {Stangeland, Tord Sture and Köhler, Andreas and Mæland, Steffen and Ramírez Rivera, Adín},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {39},
  year      = {2026}
}
```
{% endraw %}
