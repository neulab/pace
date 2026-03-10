\section{Introduction}

% Opening: Set the stage with the importance of agentic benchmarks
The rapid advancement of large language models (LLMs) has given rise to a new class of \emph{agentic systems}---tool-using AI agents that interact with complex environments to solve multi-step tasks such as software engineering, web navigation, and autonomous research~\citep{TODO_swebench,TODO_webarena,TODO_gaia}. 
Evaluating these systems has become critical for guiding model development and understanding capability frontiers.
Benchmarks like SWE-Bench~\citep{TODO_swebench}, WebArena~\citep{TODO_webarena}, and GAIA~\citep{TODO_gaia} have emerged as the de facto gold standards, testing whether agents can successfully resolve real-world GitHub issues, navigate interactive websites, or complete multi-modal reasoning tasks.

% Problem: These benchmarks are expensive and slow
However, these agentic benchmarks suffer from a fundamental tension between \emph{evaluative fidelity} and \emph{practical cost}.
Running a single model through SWE-Bench requires spinning up isolated Docker containers for each of hundreds of repository instances, executing multi-turn agent loops with tool calls, and verifying solutions against test suites---a process that can consume thousands of API calls and several hours of wall-clock time per model.
WebArena demands live browser automation across dozens of web applications, introducing additional infrastructure complexity and flakiness.
The result is that comprehensive agent evaluation remains prohibitively expensive: a single SWE-Bench run can cost hundreds of dollars in API fees alone, and full benchmark sweeps across multiple model variants can take days to complete.
This bottleneck severely limits the cadence of model iteration, makes broad hyperparameter searches impractical, and hinders reproducible comparison across research groups.

% Gap: No good middle ground exists
Existing solutions to this problem are unsatisfying.
Researchers often resort to evaluating on small, ad-hoc subsets of these benchmarks, but such subsets are typically chosen without principled criteria and may not preserve the difficulty distribution or discriminative power of the full benchmark.
Alternatively, some rely on ``proxy'' metrics like perplexity or single-turn accuracy on tangentially related tasks, but these correlate poorly with actual agent performance on complex, multi-step problems.
The field lacks a systematic methodology for constructing \emph{cheap, fast evaluation suites that reliably predict expensive agentic benchmark outcomes}.

% Our approach: Correlation-driven proxy benchmark construction
In this work, we address this gap by proposing a \textbf{correlation-driven framework for constructing proxy benchmarks}.
Our key insight is that agentic performance can be predicted from performance on carefully selected \emph{atomic evaluation items}---simple, fast-to-evaluate tasks that capture the underlying difficulty factors of complex agent benchmarks without requiring full agent rollouts.
Rather than hand-picking such items based on intuition, we \emph{directly optimize} for high correlation with target benchmark scores across a diverse set of models.

% Method overview
Our method proceeds in three stages.
First, we assemble a large \textbf{candidate pool} of atomic evaluation items drawn from existing LLM benchmarks and agent trajectories.
These items span diverse capabilities relevant to agentic tasks: next-step action prediction, tool-choice accuracy, error detection and recovery, UI element grounding, code localization, instruction following, and logical reasoning.
Crucially, each item is cheap to evaluate---requiring only a single forward pass or at most a few API calls.

Second, we formulate proxy benchmark construction as a \textbf{subset selection problem} under explicit budget constraints.
Given per-model scores on the candidate pool and on the target agentic benchmark, we select a compact subset of $k$ items that maximizes the Pearson or Spearman correlation between the proxy score (mean accuracy on selected items) and the target benchmark score.
We employ a greedy-with-swaps optimization procedure with multiple random restarts, combined with bootstrap stability selection across held-out models to ensure generalization.

Third, we evaluate the resulting proxy benchmarks through \textbf{systematic experiments} targeting SWE-Bench and related agentic benchmarks.
We compare our optimized proxies against random subsets and heuristic baselines (e.g., selecting hardest items, or items from a single benchmark), and assess generalization via cross-validation and held-out model evaluation.

% Key findings
Our experiments yield several key findings:
\begin{itemize}
    \item A proxy benchmark of roughly \textbf{100 items} can achieve meaningful correlation ($r > 0.8$) with SWE-Bench rankings at approximately \textbf{1/100th the evaluation cost}---reducing multi-hour, multi-hundred-dollar evaluations to minutes and cents.
    \item The accuracy--cost tradeoff is \textbf{smooth and predictable}: as the proxy budget $k$ increases, predictive fidelity improves monotonically, allowing practitioners to choose their desired operating point.
    \item Optimized subsets \textbf{significantly outperform} random and heuristic baselines at all budget levels, demonstrating that principled selection matters.
    \item The selected items exhibit \textbf{interpretable structure}, drawing disproportionately from code-related reasoning tasks and error-detection items---capabilities that plausibly underlie agentic software engineering performance.
\end{itemize}

% Contributions
In summary, our contributions are:
\begin{enumerate}
    \item A \textbf{general framework} for constructing proxy benchmarks that predict agentic performance from cheap atomic evaluations, formalized as correlation-maximizing subset selection.
    \item A \textbf{practical methodology} combining greedy optimization, bootstrap stability, and held-out validation to select robust, generalizable proxy subsets.
    \item \textbf{Empirical evidence} that small ($\sim$100 item) proxies can reliably approximate expensive SWE-Bench-style evaluations, enabling fast iteration during model development.
    \item A \textbf{reusable benchmark artifact}: the optimized proxy suite itself, which can be adopted by the community for rapid agent evaluation.
\end{enumerate}

% Paper outline
The remainder of this paper is organized as follows.
Section~\ref{sec:related} reviews related work on benchmark efficiency and correlation-based evaluation.
Section~\ref{sec:method} details our proxy construction methodology.
Section~\ref{sec:experiments} presents experimental results on SWE-Bench and related targets.
Section~\ref{sec:analysis} provides analysis of selected items and ablation studies.
Section~\ref{sec:discussion} discusses limitations and future directions, and Section~\ref{sec:conclusion} concludes.