# minillm
> 大模型实践之路，纸上学来终觉浅，觉知此事要躬行！我只是一个小学生，一个喜欢向着光奔跑小学生！
* 此开源项目目的旨在深入细致地了解大语言模型的核心原理，人人都可以从0到1训练并了解大语言模型。而不是像黑盒一样，仅仅会使用它，**了解它，才能更高的运用它！**
* 项目没有使用任何像LamaFactory大语言模型训练框架，从0到1仅使用Pytorch纯手工打造，涵盖预训练、监督微调（SFT）、LoRA微调、偏好优化（DPO）、强化学习训练（RL）等大语言模型各个阶段。

# Update Log
- [X] 2025.12.12 Pretrain 训练框架check并编写完成
- [X] 2025.12.11 scripts推理脚本完成，包含自实现generate和使用transformers包实现的inference
- [X] 2025.12.09 PV Cache机制实现完成
- [X] 2025.11.28 Attention及相关变体实现

# Datasets
- [ ] 待完善补充

# Appendix
完成过程参考了众多开源项目，感谢[minimind](https://github.com/jingyaogong/minimind)项目，jingyaogong大佬实现的过程深入检出，使我受益良多，再次感谢！Respect！