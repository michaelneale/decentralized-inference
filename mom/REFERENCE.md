In terms of API calls, the Mixture-of-Agents (MoA) framework functions as a multi-stage orchestration where the outputs of one set of calls become the inputs for the next. This process transforms a single user query into a series of parallel and sequential requests to various open-source models.
Here is the breakdown of how these API calls are structured across different MoA implementations:
1. The Multi-Layer API Sequence
In a standard MoA setup, the process is divided into layers of inference:
Layer 1 (Parallel Generation): The system initiates multiple simultaneous API calls to different open-source LLM agents (e.g., Llama-3, Qwen, Mixtral) using the original user prompt
.
Layer 2 and Beyond (Refined Synthesis): For the next layer, the system makes a new set of API calls. However, the payload for these calls is modified: it includes the original user prompt plus the raw text outputs from all Layer 1 agents, provided as "auxiliary information"
.
Final Aggregator Call: A final API call is usually made to a "proposer" or "aggregator" model that synthesizes all previous responses into a single, high-quality output
.
2. Efficiency-First Calls (RouteMoA)
To avoid the massive cost of calling every model in a large pool for every query, the RouteMoA framework adds a "pre-inference" step:
Lightweight Scorer Call: Before any heavy LLMs are engaged, a "lightweight scorer" (which can be a very small model or local function) predicts which models will perform best for that specific query
.
Selective API Calls: Instead of calling the entire fleet, the system only sends API requests to a high-potential subset of models, reducing computational costs by approximately 89.8%
.
3. Deliberative and Iterative Calls (NSED)
More advanced frameworks like N-Way Self-Evaluating Deliberation (NSED) treat the API interaction as a Macro-Scale Recurrent Neural Network (RNN):
Dynamic Brokerage: A runtime engine binds specific API endpoints to "functional roles" (e.g., one model acts as a "critic," another as a "coder") based on live cost and performance data
.
The Consensus Loop: The "consensus state" from one round of calls is looped back into the next round through a semantic forget gate, allowing the models to iteratively refine the answer without needing to re-send the entire conversation history every time
.
Quadratic Voting: A final activation function (like Quadratic Voting) is applied to the results of these calls to determine a non-linear consensus, ensuring that the most logically sound response prevails
.
4. Sequential and Self-MoA Calls
If you are using Self-MoA, the API structure changes slightly to focus on a single high-performing model:
Diverse Sampling: The system makes multiple API calls to the same top-performing model but uses different sampling parameters (like temperature or top-p) to generate diverse drafts
.
On-the-fly Aggregation: In the Sequential Self-MoA version, the system can aggregate these outputs "on-the-fly" over multiple rounds of calls, which is as effective as calling all outputs at once but more flexible for real-time applications
.
By organizing API calls this way, these frameworks allow smaller models (under 20B parameters) to collectively achieve the reasoning depth of 100B+ parameter models
.
So what can be done to make use of parallel calls to limit the latency of sequential

To mitigate the latency inherent in sequential, multi-layered Mixture-of-Agents (MoA) architectures, several strategies are employed to maximize parallel execution and trim the sequential chain.
1. Parallel Generation within Layers
The foundational MoA architecture uses parallelization at every layer to limit bottlenecks. Instead of one model working after another in a single line, each layer comprises multiple LLM agents that generate their responses simultaneously
. This ensures that the time taken for a single layer is only as long as its slowest individual agent, rather than the sum of all agents in that layer
.
2. Pre-Inference Dynamic Routing
To avoid the latency of running an entire fleet of models sequentially or even in large parallel batches, RouteMoA introduces a lightweight scorer that performs an initial screening
.
Predictive Selection: The scorer predicts model performance based on the query before any inference happens, narrowing the candidates to a "high-potential subset"
.
Latency Reduction: By only making parallel calls to a relevant minority of models rather than the whole pool, this framework has been shown to reduce latency by 63.6% in large-scale model environments
.
3. Adaptive Early Stopping
Advanced frameworks like Attention-MoA use an Adaptive Early Stopping Mechanism to prune the sequential process
.
Stopping the Chain: While MoA typically passes information through multiple sequential layers, this mechanism evaluates the quality of the refined logic in real-time
.
Efficiency: If the system determines the response is already optimized, it terminates the sequential chain early, preventing unnecessary late-stage API calls and significantly improving computational efficiency
.
4. Parallel vs. Sequential Aggregation (Self-MoA)
Research into Self-MoA distinguishes between two ways of handling multiple model outputs:
Parallel Aggregation: The standard method is to aggregate all model outputs at once, which uses parallel API calls to gather diverse drafts simultaneously before a final synthesis
.
Sequential Flexibility: Interestingly, a "sequential version" of Self-MoA has been developed that aggregates outputs "on-the-fly" over multiple rounds
. While this sounds slower, the sources note it is as effective as aggregating all at once, providing a flexible alternative for environments where massive parallel bursts might hit rate limits or network congestion
.
5. Hardware Arbitrage and RNN-like States
The N-Way Self-Evaluating Deliberation (NSED) protocol optimizes latency by treating the process as a Macro-Scale Recurrent Neural Network (RNN)
. It uses a Dynamic Expertise Broker that binds heterogeneous models to functional roles based on live telemetry and cost constraints
. This allows the system to intelligently manage parallel "peer review" calls across an orchestration fabric, ensuring that high-latency sequential steps are only used when the "consensus state" requires deep iterative refinement
.
What about structured output like tool calling

While the sources do not explicitly use the term "tool calling" in a programmatic JSON sense, they describe several mechanisms that directly enable Mixture-of-Agents (MoA) to handle high-logic, structured tasks such as coding and functional reasoning through functional specialization and peer-mediated correction.
Here is how these frameworks manage structured outputs and complex logic:
1. Assignment of Functional Roles
Advanced MoA frameworks like N-Way Self-Evaluating Deliberation (NSED) do not treat all agents as identical. Instead, they use a Dynamic Expertise Broker that binds different models to specific "functional roles" based on their performance telemetry
.
In a tool-calling scenario, this allows the system to assign a model known for high structural accuracy to the role of syntax validator or parameter extractor, while using others for broader reasoning
.
2. Logic Refinement and Hallucination Correction
A major challenge with structured output is "hallucinating" parameters or failing to follow syntax. Attention-MoA is specifically designed to address this by facilitating Inter-agent Semantic Attention
.
This mechanism allows agents to actively correct hallucinations and refine logic by interacting with the outputs of other agents
.
If one model generates a malformed tool call, subsequent layers can identify the structural error and refine it before the final output is produced
.
3. Peer-Mediated Structural Integrity
The NSED protocol introduces an orchestration fabric for trustless N-to-N peer review
.
This creates a consensus state—functioning like a Macro-Scale Recurrent Neural Network (RNN)—where models iteratively deliberate on the output
.
This peer-mediated process has been shown to reduce "sycophancy" and improve alignment, which is critical when a model must strictly adhere to a specific output schema rather than just generating creative text
.
4. Performance on Structured Benchmarks
The effectiveness of these frameworks on structured tasks is supported by their performance on technical benchmarks:
LiveCodeBench: NSED demonstrated that ensembles of small, consumer-grade models (under 20B parameters) can match or exceed the performance of massive 100B+ parameter models on this coding-focused benchmark
.
FLASK Benchmark: Attention-MoA dominated in 10 out of 12 capability categories on FLASK, a benchmark that measures fine-grained model skills, including logical robustness
.
MATH and CRUX: The Self-MoA approach showed an average of 3.8% improvement across reasoning-heavy benchmarks like MATH and CRUX, suggesting that even self-collaboration improves the reliability of logical outputs
.
In summary, although the sources focus on the architectural "how," they indicate that MoA handles structured requirements by breaking the task into roles, using semantic attention to fix structural errors, and employing iterative peer review to ensure the final output meets the required logical or syntactic standards
.
How does it judge and score things and choose. Use can you use pseudo code and real model names

The Mixture-of-Agents (MoA) framework judges, scores, and chooses models through a multi-stage process that combines predictive screening, peer-mediated assessment, and mathematical consensus.
1. Pre-Inference: The "Knapsack" Selection
Before any heavy models are called, newer frameworks like RouteMoA and N-Way Self-Evaluating Deliberation (NSED) perform a screening step to avoid the cost of running every available model
.
Lightweight Scorer: A small, fast model (often a classifier or a tiny LLM) predicts the "coarse-grained performance" of models in the pool based solely on the user's query
.
The Knapsack Problem: NSED treats model selection as a variation of the Knapsack Problem, where a Dynamic Expertise Broker chooses the best combination of models (e.g., Llama-3-8B, Qwen-2-7B, or Mistral-7B) that fits within the user's cost and latency constraints while maximizing expected performance
.
2. Posterior Assessment: The "Mixture of Judges"
Once the selected models generate initial drafts, the system must score the quality of those drafts to determine which should influence the final answer.
Self- and Cross-Assessment: RouteMoA uses a "mixture of judges" to refine the initial scores. This involves models looking at their own outputs (self-assessment) and the outputs of their peers (cross-assessment) to provide posterior correction
.
Inter-agent Semantic Attention: In the Attention-MoA framework, agents do not just "vote"; they use semantic attention to facilitate deep interaction, allowing them to actively identify and correct logic errors or hallucinations in each other's work
.
3. Final Choice: Quadratic Voting and Ranking
To reach a final decision, the system aggregates the scores and outputs into a single response.
Quadratic Voting: NSED utilizes a Quadratic Voting activation function. This creates a non-linear consensus, meaning that a strong agreement among high-performing "expert" agents can outweigh a simple majority of weaker models, ensuring robust logic
.
Model Ranking Mechanism: RouteMoA applies a final ranking that balances performance, cost, and latency
. This allows the system to choose a "proposer" model (like a larger Llama-3-70B) to synthesize the final answer based on the highest-ranked drafts.
Logic Workflow (Pseudo-code)
# Mixture-of-Agents Scoring and Selection Logic

def moa_orchestrator(user_query, model_pool):
    # 1. PRE-INFERENCE SCREENING (RouteMoA / NSED)
    # Predict performance and solve for cost/latency constraints
    candidates = scorer.predict_performance(user_query, model_pool)
    selected_agents = dynamic_expertise_broker.solve_knapsack(
        candidates, cost_limit=0.05, max_latency=2.0
    ) # Binds models like Qwen-7B or Llama-8B to functional roles [1, 2]

    # 2. PARALLEL GENERATION (Layered MoA)
    drafts = parallel_inference(selected_agents, user_query) # [5]

    # 3. SCORING & REFINEMENT (Attention-MoA / RouteMoA)
    # Refine logic using Inter-agent Semantic Attention [3]
    refined_logic = semantic_attention_refine(drafts)
    
    # Posterior scoring via self/cross assessment
    scores = mixture_of_judges.assess(refined_logic) # [2]

    # 4. FINAL CONSENSUS (NSED)
    # Apply Quadratic Voting for non-linear logic consensus [1]
    consensus_state = quadratic_voting(scores, refined_logic)
    
    # 5. AGGREGATION
    # Proposer model (e.g. Llama-3-70B) synthesizes final response
    final_response = aggregator.synthesize(consensus_state)
    
    return final_response
By using these methods, these frameworks enable ensembles of small open-source models (under 20B parameters) to outperform massive proprietary systems like GPT-4.1 and Claude-4.5-Sonnet
.
Today • 3:34 AM
How is score calculated? What is functional model how to do know what is what

In the Mixture-of-Agents (MoA) framework, scoring and model selection are handled through a tiered process that moves from predictive screening to real-time deliberative consensus.
1. How the Score is Calculated
The "score" of a model or a specific response is determined across three distinct phases:
Pre-Inference Prediction: A lightweight scorer (a small model or classifier) analyzes the user query and predicts the coarse-grained performance of each available model
. This allows the system to identify a "high-potential subset" without wasting resources on every model in the pool
.
Posterior Correction (Mixture of Judges): Once the selected models generate initial responses, a mixture of judges refines the initial scores
. These judges perform self-assessment (evaluating their own logic) and cross-assessment (evaluating their peers' logic) based on the actual text outputs
.
Quadratic Voting Activation: To reach a final consensus, the system uses a Quadratic Voting activation function
. Unlike a simple average, this mathematical mechanism creates a non-linear consensus, weighting the logic of "expert" models more heavily if there is strong agreement, which reduces sycophancy and ensures logical robustness
.
2. Functional Models: Knowing "What is What"
A functional model refers to the specific "role" a model plays within the composite system (e.g., proposer, critic, or aggregator). The system decides "what is what" through a Dynamic Expertise Broker
.
The Knapsack Problem: The broker treats model selection as a variation of the Knapsack Problem
. It evaluates the "weight" (computational cost/latency) and "value" (predicted performance) of heterogeneous checkpoints—such as Llama-3-8B, Qwen-2-7B, or Mistral-7B
.
Telemetry-Driven Binding: The system binds these different checkpoints to functional roles at runtime based on live telemetry and cost constraints
. For example, a model with high coding accuracy on LiveCodeBench might be dynamically assigned the role of "code validator" for a technical query
.
Logic Workflow (Pseudo-code)
# MoA Scoring and Functional Role Selection

def calculate_moa_score(user_query, model_fleet):
    # 1. INITIAL SCREENING (RouteMoA)
    # Predicted performance vs. Cost/Latency
    predicted_performance = lightweight_scorer.predict(user_query, model_fleet) # [1]
    
    # 2. DYNAMIC BROKERAGE (NSED)
    # Select best 'Functional Models' by solving the Knapsack Problem
    # e.g., mapping 'Qwen-2-7B' to 'Critic' and 'Llama-3-8B' to 'Proposer'
    selected_functional_roles = dynamic_broker.solve_knapsack(
        predicted_performance, 
        max_cost=0.10, 
        max_latency=1.5
    ) # [2]

    # 3. POSTERIOR ASSESSMENT (Mixture of Judges)
    # Models generate outputs and judge each other
    drafts = parallel_inference(selected_functional_roles, user_query)
    refined_scores = mixture_of_judges.cross_assess(drafts) # [1]

    # 4. NON-LINEAR CONSENSUS (NSED)
    # Apply Quadratic Voting to determine the strongest logic
    final_consensus_state = quadratic_voting(refined_scores, drafts) # [2]

    # 5. AGGREGATION
    # Final response synthesized by top-tier model (e.g., Llama-3-70B)
    return aggregator.synthesize(final_consensus_state)
By using these methods, small open-source ensembles (under 20B parameters) achieve a "hardware arbitrage" that allows them to outperform massive proprietary models like GPT-4.1 and Claude-4.5-Sonnet
.
Please go into detail for each step. Include how things work. And also imagine a tool call will have to be made

The Mixture-of-Agents (MoA) workflow is an advanced orchestration process that shifts LLM development from simple parameter scaling to inference-time collaboration
. In a scenario involving a complex tool call, the process follows five distinct, data-driven steps to ensure the final output is structurally sound and logically accurate.
1. Pre-Inference: Dynamic Brokerage and Selection
Before any major computation occurs, the system must decide which models are best suited for the task while adhering to cost and latency constraints
.
How it works: A lightweight scorer performs initial screening by predicting "coarse-grained performance" directly from the user's query without running a full inference
.
The Knapsack Problem: A Dynamic Expertise Broker treats model selection as a variation of the Knapsack Problem
. It binds heterogeneous checkpoints—such as Llama-3-8B or Qwen-2-7B—to specific functional roles based on live telemetry
.
Tool Call Context: The broker might assign a model known for high structural accuracy (e.g., Mistral-7B) the role of "Parameter Extractor" to identify the specific arguments required for the tool.
2. Execution Layer: Parallel Generation
Once the models are selected, they begin generating responses in a dense, layered topology
.
How it works: In the first layer, multiple agents generate their initial drafts simultaneously. If Self-MoA is used, the system may generate multiple diverse outputs from the single top-performing model to maintain high average quality
.
Collective Expertise: These agents use each other's outputs as "auxiliary information," effectively treating the first round of drafts as context for the next
.
Tool Call Context: Parallel agents (e.g., various 8B-parameter models) simultaneously draft the JSON schema for the tool call. This provides the system with multiple "draft" versions of the required parameters.
3. Deliberation: Semantic Attention and Logic Refinement
To prevent a "telephone game" effect in deep layers, the system uses advanced interaction mechanisms to refine the logic
.
How it works: Attention-MoA introduces Inter-agent Semantic Attention, which facilitates deep interaction rather than simple text concatenation
. This allows models to actively correct hallucinations and refine logic
.
Residual Synthesis: To prevent information degradation as the data passes through layers, Inter-layer Residual Modules and Adaptive Early Stopping Mechanisms are used to maintain efficiency
.
Tool Call Context: If one model hallucinates a non-existent API parameter, the semantic attention mechanism allows other models to identify the logical inconsistency and correct the JSON structure before it proceeds
.
4. Assessment: Non-Linear Consensus via Quadratic Voting
Before a final answer is chosen, the system must mathematically determine which logical path is the most robust
.
How it works: A Mixture of Judges performs both self-assessment and cross-assessment of the refined drafts
.
Quadratic Voting: The N-Way Self-Evaluating Deliberation (NSED) protocol uses a Quadratic Voting activation function
. This creates a non-linear consensus, ensuring that the most rigorous logic—often from the most specialized "expert" agents—prevails over a simple majority
.
Tool Call Context: The system uses Quadratic Voting to "vote" on the final values for the tool's arguments. This reduces "sycophancy" and ensures that the final parameters are precisely aligned with the user's intent
.
5. Final Synthesis: The Aggregator
The final step involves a high-tier model synthesizing the collective deliberation into a single, executable response
.
How it works: A "proposer" or aggregator model—typically a larger open-source model like Llama-3-70B—takes the final consensus state and produces the ultimate output
.
Performance: This collaborative approach allows ensembles of small (less than 20B) consumer-grade models to match or exceed the performance of 100B+ parameter models like GPT-4.1 and Claude-4.5-Sonnet
.
Tool Call Context: The aggregator outputs the finalized, verified tool-call JSON, ready for execution.
By treating the entire process as a Macro-Scale Recurrent Neural Network (RNN) with a semantic forget gate, the MoA framework allows for iterative refinement without the need for massive, localized VRAM scaling
.
How does it know verified tool call schema

The Mixture-of-Agents (MoA) framework does not rely on a single hard-coded validator to know a tool-call schema; instead, it "knows" and verifies the schema through collective deliberation, functional role assignment, and iterative peer review.
Based on the sources, here is the detailed breakdown of how a verified schema is maintained across the architecture:
1. Functional Specialization via the Dynamic Expertise Broker
The system begins by identifying models within its pool that have high structural and logical accuracy.
How it works: The Dynamic Expertise Broker treats model selection as a Knapsack Problem, matching specific heterogeneous checkpoints (like Llama-3-8B or Qwen-2-7B) to functional roles based on their performance telemetry
.
The Schema "Knowledge": While the user prompt contains the tool-call schema, the broker ensures that a model with high scores on benchmarks like LiveCodeBench or FLASK is assigned to a "Validator" or "Structural Critic" role
. This model is essentially the "expert" that knows how to interpret the schema provided in the prompt.
2. Deep Semantic Correction via Attention-MoA
Standard text-to-text passing can lead to a "telephone game" where structural details are lost. Attention-MoA prevents this.
Inter-agent Semantic Attention: Instead of just reading text, models use semantic attention to interact deeply with the outputs of other agents
.
Hallucination Correction: If a "proposer" model generates a tool call with a hallucinated parameter (one not in the schema), the "critic" models use semantic attention to highlight this specific logical error
. This allows the system to actively correct hallucinations and refine logic in real-time
.
3. Iterative Refinement (Macro-Scale RNN)
The framework treats the entire collaboration as a Macro-Scale Recurrent Neural Network (RNN) to ensure the final output is structurally sound.
The Consensus Loop: The "consensus state" of the agents loops back through a semantic forget gate
. This allows the models to iteratively "clean" the tool call.
Feedback-Driven State Update: If the group identifies that the current draft of the tool call does not match the required schema, the feedback is used to update the state in the next round of deliberation until a verified version is reached
.
4. Verification through N-to-N Peer Review
The most critical verification step is the orchestration fabric for trustless N-to-N peer review
.
Mixture of Judges: RouteMoA uses a "mixture of judges" to perform self- and cross-assessment of the generated outputs
.
Trustless Verification: Multiple agents act as "peers" who review the tool call's JSON structure. Because these models are from different families (heterogeneous agents), they are less likely to share the same hallucination biases, leading to peer-mediated correction that reduces sycophancy and improves alignment with the schema
.
5. Non-Linear Consensus (Quadratic Voting)
To ensure the "verified" call is the one that actually leaves the system, a final mathematical filter is applied.
Quadratic Voting: The system uses a Quadratic Voting activation function to determine a non-linear consensus
.
Expert Priority: If the models assigned to the "structural validator" roles strongly agree that a tool call matches the schema, their "vote" carries non-linear weight. This ensures that a single robust, verified logical path prevails over several slightly flawed versions
.
Summary Logic for a Tool Call:
Broker selects models based on their ability to handle structured code (e.g., LiveCodeBench performance)
.
Layer 1 agents draft the tool call based on the schema provided in the auxiliary information of the prompt
.
Attention-MoA identifies structural discrepancies (hallucinations) in those drafts using Inter-agent Semantic Attention
.
N-to-N Peer Review verifies the syntax against the original schema instructions
.
Quadratic Voting ensures the final, verified JSON is the output that achieves consensus
.
By using this "hardware arbitrage," an ensemble of small, consumer-grade models (under 20B) can achieve a level of structural verification that matches or exceeds that of 100B+ parameter models like GPT-4.1
.
can have a tiny bit detail on what nsed does and how to score. And how to use a model for the tool calling specifically

The N-Way Self-Evaluating Deliberation (NSED) protocol is a runtime architecture that transforms a group of distinct expert agents into a single "emergent composite model"
. Unlike static systems, NSED treats the collaboration process as a Macro-Scale Recurrent Neural Network (RNN), where the collective "consensus state" loops back through a semantic forget gate to allow for iterative refinement without requiring massive increases in memory (VRAM)
.
How NSED Scores and Chooses Models
The scoring and selection process in these advanced frameworks occurs in three distinct layers to balance performance with efficiency:
The Knapsack Selection (Pre-Inference): Instead of using a fixed set of models, NSED uses a Dynamic Expertise Broker
. This engine treats model selection as a variation of the "Knapsack Problem," where it calculates the best combination of models (e.g., Llama-3-8B, Qwen-2-7B) to use based on live telemetry, cost constraints, and latency limits
.
Predictive Scoring: To avoid the high cost of running every model, RouteMoA employs a lightweight scorer that predicts the "coarse-grained performance" of models based only on the user's query before any inference occurs
.
Posterior Correction (Mixture of Judges): Once models generate drafts, a "mixture of judges" performs self- and cross-assessment
. This provides a "posterior correction," where agents evaluate their own and their peers' logic to refine the initial scores
.
Quadratic Voting (The Final Decision): NSED reaches a final choice using a Quadratic Voting activation function
. This mathematical mechanism creates a non-linear consensus, meaning the system doesn't just take a simple average; it gives significantly more weight to "expert" agents that show strong logical agreement, which effectively reduces sycophancy
.
Using a Model Specifically for Tool Calling
In a tool-calling scenario, the framework does not treat the LLM as a general text generator but rather binds it to a functional role
. Here is how you use a model for a specific tool call within this architecture:
Role Binding: The Dynamic Expertise Broker identifies models with high structural accuracy (verified by benchmarks like LiveCodeBench) and assigns them the specific role of "Schema Validator" or "Parameter Extractor"
.
Semantic Attention for Schema Integrity: For a tool call to work, the JSON syntax must be perfect. Attention-MoA uses Inter-agent Semantic Attention to allow models to "interact deeply"
. If one model generates a tool call with a hallucinated parameter, the other models use this semantic interaction to actively correct the hallucination and refine the logic before the final output
.
Iterative Refinement of the Call: Because NSED functions like an RNN, if the initial tool call draft is malformed, the "consensus state" loops back through the semantic forget gate
. The models then use the feedback from the "peer-review" fabric to update the state and produce a corrected, verified tool call in the next iteration
.
By using this "hardware arbitrage," an ensemble of small, consumer-grade models (under 20B parameters) can achieve the structural and logical precision required for complex tool calling, often matching or exceeding the performance of massive 100B+ parameter models
.
