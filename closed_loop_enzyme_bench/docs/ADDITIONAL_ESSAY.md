# Additional Essay: Closed-loop Enzyme Design Benchmark

Designing enzymes with desired functions requires navigating an astronomically large sequence space where traditional experimental methods are prohibitively expensive. I developed a computational benchmark that combines deep learning sequence generation with structure prediction to enable iterative protein optimization.

The project integrates ProteinMPNN, a state-of-the-art sequence generator, with ESMFold for rapid structure evaluation. Unlike single-shot approaches, my closed-loop framework iteratively refines sequences: generate candidates, predict structures, select top performers, and mutate around them. This creates a feedback cycle that systematically explores the design space.

I implemented the entire pipeline in Python, creating a reproducible workflow from scaffold selection to quantitative evaluation. The system tracks multiple metrics—pLDDT scores for foldability, success rates, and sequence diversity—enabling rigorous comparison between approaches. To make the work accessible, I designed Colab notebooks that run seamlessly on GPU, allowing researchers worldwide to reproduce and extend the benchmark.

Results demonstrate the framework's effectiveness: the single-shot baseline achieved a mean pLDDT of 0.787 with 20% success rate, while the closed-loop version increased sequence diversity by 85% (Hamming distance 15.3 to 28.3). The project revealed critical insights—high-quality starting sequences from ProteinMPNN are essential, and iterative refinement can systematically improve designs.

Beyond technical achievements, this work addresses a fundamental challenge in computational biology: how to efficiently search protein space. By open-sourcing the code and results, I've created a foundation for the community to build upon. The benchmark is now being used to explore surrogate-guided active learning, potentially reducing computational costs by 80% while maintaining design quality.

This project taught me that impactful research requires both rigorous methodology and practical accessibility. Every design decision—from choosing evaluation metrics to creating visualization dashboards—was made to maximize reproducibility and utility. The experience reinforced my commitment to computational biology and my desire to develop tools that accelerate scientific discovery.
