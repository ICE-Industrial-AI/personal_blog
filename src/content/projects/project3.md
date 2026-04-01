---
title: "KG melding with LLMs"
description: "The project develops a graph-based representation of the physical world, mapping real objects to nodes with relevant functional attributes. This enables precise task planning and graph-based reasoning about object relationships, supporting exact predictions beyond the capabilities of conventional LLMs. Knowledge graphs are well suited for modeling and querying complex relationships, and this approach extends them by encoding functional properties in node and edge attributes. The method is illustrated in two key domains: robotics, where graphs are already central to task planning, and engineering design, which has recently gained attention through LLMs and generative AI. Despite this potential, challenges remain in how systems are described and represented."
pubDate: "Mar 31 2026"
heroImage: "/personal_blog/spark/SparkPoster.png"
badge: "Starting"
tags: ["AI", "LLMs", "Knowledge Graphs", "Feedback"]
category: "AI Research"
featured: true
---

# K-GraphLink: Knowledge-Graphs Linking physical world and code for task planning

# Abstract
The project envisions the development of a graph representation that abstracts the physical world but maps it to a graph with relevant and functional attributes. This representation allows to create precise task planning based on this graph model, with the connections between the real objects, represented as graph nodes, allowing exact predictions to be made based on graph reasoning, which is not possible with conventional Large Language Models (LLMs). Indeed, knowledge graphs are among the most effective methods for mapping and querying complex relationships, and this method is also employed in this context, with the representation of functional properties in node and edge attributes representing a novel innovation. The method is utilized as an illustration in two pertinent domains. The former pertains to robotics, where graphs already play a signiﬁcant role in task planning; the latter focuses on engineering design, an area that has recently gained signiﬁcant relevance due to LLMs (large language models) and generative AI. However, it should be noted that the ﬁeld still faces challenges in terms of description and representation of systems.


# Aim of the Project

Despite the rapid development of neural networks, in particular in the field of solving tasks with LLMs, shortcomings subsist, especially in the domain of engineering design and geometric representation learning e.g. for task planning in robotics where complex multiple physical object relation and interaction is crucial. The associative nature of transformers models, due to the lack of consistency, makes inference in rule systems unreliable (C1: Challenge 1 - Consistency). Chopping words into tokens for transformation into latent space disrupts essential numeric representations, e.g. the representation of numbers and the exact retrieval of numerical geometric and physical information (C2: Challenge 2 - Implicit Relation Constraints). LLM planning methods like Chain-of-Though [1]  improve LLMs’ reasoning, laying the foundation for models like GPT-o1. However, there is still a leak of accurate physical reasoning and information transfer over multiple thinking and reasoning steps and long context inputs due to the purely token based sequential representation and applied attention mechanism (C3: Challenge 3 – Planning). The following use cases show system models requiring a high level of detail for solving complex reasoning and planning tasks.

## Use Case 1: Robotics
“Imagine a patient being handed a book by a nursing robot. There is a cup of hot water on the book. It is not sufficient for the objects themselves to be recognized in the room. It is also necessary to determine the relations and relationships between the objects and their physical properties. For example, the relationship between the book and the cup must be modeled for this task. Once this relationship is known, the subtasks can be planned and executed.”

![Figure 1](/personal_blog/spark/HotCupOverBook.png)



## Use Case 2: Engineering Design Automation (EDA)
“In engineering design, systems comprise multiple interconnected objects, each exhibiting geometric and physical interdependencies. The configuration of these objects is thus contingent on geometric and physical properties, in addition to adhering to specific functional requirements. A notable illustration of this complexity arises in the design of an electrical circuit board, a highly intricate problem that necessitates the consideration of manufacturing, electromagnetic, and customer-specific requirements. It is therefore essential to model these dependencies, as is commonly practiced in the acquisition and planning of robotics tasks.”


A closer inspection of both use cases reveals considerable synergies: existing methods can be transferred accordingly, and new methods can be applied to both domains. 


This project aims at designing and testing a workflow that integrates graphs with multimodal embeddings in a simulated environment. Based on the above-mentioned problems of complex reasoning, many approaches for tasks such as planning and retrieval use graphs, more specifically knowledge graphs. A further component for physical planning is needed in the form of geometrical deep learning embedded in a neural graph to account in the inference with different multi-modal constraints.

This can solve many problems in robotics, such as giving functional semantics to pixels assigned to objects, e.g. so that we can infer if the represented object is graspable or movable. In use case 1, using a graph would allow creating a digital twin model, namely a simulated environment, and allow verifying that the next selected action leads to the desired outcome and the plan toward the goals, before moving any real parts. For example, when a robot lifts the cup, it needs to plan with functional and physical constraints, like weight and temperature.


# Research Questions:

- RQ1: What knowledge and representations can be effectively transferred from robotics tasks and scene segmentation to EDA, and what adaptations are necessary to account for domain-specific requirements?
- RQ2: How can CAD models be efficiently transformed into neural GBR, capturing hierarchical relationships between elements and sub-elements using graphs and sub-graphs (C1 & C2)?
- RQ3: How can the GPF multi-modal embedding graph partitioning and retrieval be realized (C2)? 
- RQ4: How can the analysis of the simulations be automatized to further improve the planned action/design, enhancing the final planning (C3)?