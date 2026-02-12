---
title: "Combinatorial Optimization on Graphs"
description: "Many central decision and optimization tasks in science and engineering can be effectively modeled on graphs. Whether it is allocating resources in complex communication networks, scheduling jobs under conflict constraints, clustering high-dimensional data, or detecting cohesive substructures in social networks, the underlying mathematical structure is often a graph. A large class of these tasks falls under the umbrella of combinatorial optimization (CO). In this article, we will explore five fundamental graph combinatorial optimization problems. While they are NP-hard, the combination of exact algorithms, approximations, and continuous relaxations provides a robust toolkit for the practitioner. Then we will show how such combinatorial optimization problems (CO) can be solved using Graph Neural Networks (GNN)."
pubDate: "Feb 12 2026"
heroImage: "/personal_blog/aikn.webp"
badge: "Latest"
---


# Combinatorial Optimization on Graphs

**Author:** *Christoph Würsch, Institute for Computational Engineering ICE* <br>
*Eastern Switzerland University of Applied Sciences OST* <br>
**Date:** 12.02.2026<br>



# Optimization on Graphs


> In this article, we will explore five fundamental graph combinatorial optimization problems. While they are NP-hard, the combination of exact algorithms, approximations, and continuous relaxations provides a robust toolkit for the practitioner. Then we will show how such combinatorial optimization problems (CO) can be solved using Graph Neural Networks (GNN).


## Table of Contents
1. [Combinatorial Optimization on Graphs](#1-combinatorial-optimization-on-graphs)
    * [1.1 Complexity and Practical Strategies](#11-complexity-and-practical-strategies)
2. [Graph Preliminaries and Notation](#2-graph-preliminaries-and-notation)
    * [2.1 Basic Definitions](#21-basic-definitions)
    * [2.2 Unified Quadratic Views](#22-unified-quadratic-views)
3. [Graph Partitioning](#3-graph-partitioning)
    * [3.1 Problem Formulation](#31-problem-formulation)
    * [3.2 Spectral Relaxation](#32-spectral-relaxation)
4. [Maximum Cut (Max-Cut)](#4-maximum-cut-max-cut)
    * [4.1 Problem Formulation](#41-problem-formulation)
    * [4.2 SDP Relaxation (Goemans--Williamson)](#42-sdp-relaxation-goemanswilliamson)
5. [Minimum Vertex Cover](#5-minimum-vertex-cover)
    * [5.1 Problem Formulation](#51-problem-formulation)
    * [5.2 Relaxations and Approximations](#52-relaxations-and-approximations)
    * [5.3 QUBO Penalty Formulation](#53-qubo-penalty-formulation)
6. [Maximum Independent Set (MIS)](#6-maximum-independent-set-mis)
    * [6.1 Problem Formulation](#61-problem-formulation)
    * [6.2 Relationship to Vertex Cover](#62-relationship-to-vertex-cover)
    * [6.3 Greedy Heuristics](#63-greedy-heuristics)
7. [Maximum Clique](#7-maximum-clique)
    * [7.1 Problem Formulation](#71-problem-formulation)
    * [7.2 Motzkin--Straus and Continuous Characterization](#72-motzkinstraus-and-continuous-characterization)
    * [7.3 Exact Search: Bron--Kerbosch](#73-exact-search-bronkerbosch)
8. [Unified View of Relaxations](#8-unified-view-of-relaxations)
    * [8.1 SDP Lifting: The Central Technique](#81-sdp-lifting-the-central-technique)
    * [8.2 Randomized Rounding](#82-randomized-rounding)
    * [Take Aways](#take-aways)
9. [Solving CO-Problems using GNN: The Data-Driven Shift](#9-solving-co-problems-using-gnn-the-data-driven-shift)
    * [9.1 Graph Neural Networks: The Engine of Reasoning](#91-graph-neural-networks-the-engine-of-reasoning)
    * [9.2 Primal Approaches: Finding Solutions Directly](#92-primal-approaches-finding-solutions-directly)
    * [9.3 Reinforcement Learning (RL) for Construction](#93-reinforcement-learning-rl-for-construction)
    * [9.4 Physics-Inspired Unsupervised Learning](#94-physics-inspired-unsupervised-learning)
    * [9.5 Dual Approaches: Enhancing Exact Solvers](#95-dual-approaches-enhancing-exact-solvers)
    * [9.6 Algorithmic Reasoning](#96-algorithmic-reasoning)
10. [Conclusion and Future Outlook](#10-conclusion-and-future-outlook)




## 1 Combinatorial Optimization on Graphs

Many central decision and optimization tasks in science and engineering can be effectively modeled on graphs. Whether it is allocating resources in complex communication networks, scheduling jobs under conflict constraints, clustering high-dimensional data, or detecting cohesive substructures in social networks, the underlying mathematical structure is often a graph. A large class of these tasks falls under the umbrella of *combinatorial optimization* (CO). In these problems, the decision variables are discrete—typically binary. For example, a node is either selected for a set or it is not; an edge is either cut or it remains intact. This discrete nature presents a fundamental challenge: standard continuous optimization tools, such as gradient-based methods, are inapplicable in their native form because gradients are not defined on discrete sets like $\{0,1\}$. Furthermore, the five classical graph problems we study in this chapter—Graph Partitioning, Maximum Cut (Max-Cut), Minimum Vertex Cover, Maximum Independent Set, and Maximum Clique—are all NP-hard in general graphs.

### 1.1 Complexity and Practical Strategies

The classification of these problems as NP-hard is significant. It implies that, assuming $\text{P} \neq \text{NP}$, we should not expect to find a polynomial-time algorithm that always returns the global optimum for every input instance. However, this theoretical hardness does not render these problems unsolvable in practice. It simply means we must adapt our expectations and strategies. Common practical strategies include:

* **Exact Methods:** Approaches such as branch-and-bound, cutting planes, and modern Integer Linear Programming (ILP) solvers can find optimal solutions. While their worst-case complexity is exponential, they are often effective on small to medium-sized instances or instances with specific structures.
* **Approximation Algorithms:** These are efficient (polynomial-time) algorithms that come with provable guarantees. For example, we will discuss the Goemans--Williamson algorithm for Max-Cut, which guarantees a solution quality of at least $87.8\%$ of the optimum.
* **Heuristics and Local Search:** These methods, while lacking worst-case guarantees, are often extremely effective and fast. They include greedy approaches and iterative improvement strategies (like flipping a variable's state to reduce energy).
* **Continuous Relaxations:** A key modern strategy, and a focal point of this chapter, is to build continuous reformulations (Spectral, Quadratic Programming, Semidefinite Programming). These relaxed problems are solvable efficiently and can be "rounded" back to valid discrete solutions.

## 2 Graph Preliminaries and Notation

Before we delve into specific problems, we must establish the mathematical framework and notation used throughout this chapter.

### 2.1 Basic Definitions

**Definition 1 (Graph)**
Let $G=(V,E,w)$ be an undirected weighted graph.
* The vertex set is $V=\{1,\dots,n\}$.
* The edge set is $E\subseteq \{\{i,j\}\,:\,i\neq j\}$.
* The weights are $w_{ij}=w_{ji}\ge 0$ for $(i,j)\in E$ (and $w_{ij}=0$ otherwise).

We will frequently utilize algebraic representations of the graph:
* $A \in \mathbb{R}^{n\times n}$ is the **adjacency matrix**, where $A_{ij}=w_{ij}$.
* $D = \mathrm{diag}(d_1,\dots,d_n)$ is the **degree matrix**, with $d_i=\sum_{j} w_{ij}$.
* $L = D-A$ is the **combinatorial Laplacian**. This matrix is positive semidefinite and plays a crucial role in spectral graph theory.

Binary decision variables appear in two common encodings in the literature. It is useful to be comfortable moving between them:
1.  **Boolean variables:** $x_i \in \{0,1\}$. These are typical in computer science and ILP formulations.
2.  **Spin variables:** $s_i \in \{-1,+1\}$. These are typical in physics (Ising models) and spectral methods.

They are linearly related by the transformations:
	$$x_i = \frac{1+s_i}{2}, \qquad s_i = 2x_i-1.$$

### 2.2 Unified Quadratic Views

Many graph CO problems can be unified under a general quadratic form. In the boolean domain, this is the **QUBO** (Quadratic Unconstrained Binary Optimization) form:

$$
\min_{x\in\{0,1\}^n} \; x^\top Q x + c^\top x.
$$

In the spin domain, this corresponds to the **Ising model**:

$$
\min_{s\in\{-1,1\}^n}\; s^\top J s + h^\top s + \mathrm{const}.
$$

These forms are important not just for mathematical elegance but because they represent the interface for new hardware accelerators (such as quantum annealers) and are standard in statistical physics for energy minimization and Gibbs distributions. To illustrate the concepts in this chapter, we will use a small running example graph $G$ with $V=\{1,2,3,4,5\}$ and unit edge weights ($w_{ij}=1$). The edges are:
$$E=\{(1,2),(1,3),(2,3),(2,4),(3,5),(4,5)\}.$$
Structurally, this graph contains a triangle on vertices $\{1,2,3\}$ and a "bridge-like" structure connecting the triangle to $\{4,5\}$.

**Example 1 (Adjacency Matrix of the Running Graph)**
With node order $(1,2,3,4,5)$, the adjacency matrix is:
$$
A=
\begin{pmatrix}
    0&1&1&0&0\\
    1&0&1&1&0\\
    1&1&0&0&1\\
    0&1&0&0&1\\
    0&0&1&1&0
\end{pmatrix}.
$$

## 3 Graph Partitioning

Graph partitioning is a cornerstone problem in parallel computing and load balancing. Consider a finite element solver where vertices represent mesh elements and edges represent adjacency (communication requirements). To distribute the simulation across multiple processors, we must partition the graph. A good partition minimizes inter-processor communication (edges cut) while keeping the computational workload balanced (equal partition sizes). This reduces overhead and speeds up parallel computation.

![Load Balancing using Graph Partitioning](/personal_blog/GraphPartitioning_LoadBalancing_small.png)

### 3.1 Problem Formulation

We define the problem for the simplest case: balanced 2-way partitioning.

**Definition 2 (Graph Partitioning (Balanced 2-Way))**
Given $G=(V,E,w)$ with $n$ even, find a partition $V=S\cup \overline{S}$ with $|S|=|\overline{S}|=n/2$ that minimizes the cut weight:
$$\mathrm{cut}(S,\overline{S}) = \sum_{i\in S,\,j\in \overline{S}} w_{ij}.$$

Let us formulate this using spin variables $s_i \in \{-1, 1\}$. We assign $s_i = +1$ if $i \in S$ and $s_i = -1$ if $i \in \overline{S}$. The indicator function for an edge being cut (endpoints having different signs) is $\frac{1-s_is_j}{2}$. Thus, the cut size is:
$$\mathrm{cut}(S,\overline{S}) = \frac12\sum_{(i,j)\in E} w_{ij}\frac{1-s_is_j}{2} = \frac14\sum_{(i,j)\in E} w_{ij}(1-s_is_j).$$
The balance constraint $|S|=|\overline{S}|$ translates to $\sum_{i=1}^n s_i = 0$. We can rewrite the sum over edges using the Laplacian quadratic form:
$$\sum_{(i,j)\in E} w_{ij}(s_i-s_j)^2 = 2 s^\top L s.$$
Minimizing the cut is therefore equivalent to minimizing $s^\top L s$ subject to the constraints.

### 3.2 Spectral Relaxation

The discrete constraints make the problem NP-hard. We perform a *spectral relaxation* by replacing the discrete vector $s \in \{-1, 1\}^n$ with a continuous vector $y \in \mathbb{R}^n$. We relax the discreteness to a spherical constraint $\|y\|_2^2 = n$ and maintain the orthogonality to the constant vector (balance). The relaxed problem is:

$$
\min_{y\in\mathbb{R}^n}\; y^\top L y \quad\text{s.t.}\quad y^\top\mathbf{1}=0,\ \|y\|_2^2=n.
$$

This is a standard Rayleigh quotient minimization on the subspace orthogonal to $\mathbf{1}$. The solution is the eigenvector associated with the **second smallest eigenvalue** of the Laplacian $L$, known as the **Fiedler vector**.

**Algorithm 1: Spectral 2-Way Partitioning (Fiedler Cut)**
1.  Compute Laplacian $L=D-A$.
2.  Compute Fiedler vector $y$: the eigenvector of $L$ corresponding to the 2nd smallest eigenvalue.
3.  Form partition by thresholding $y$ (e.g., $S=\{i: y_i\ge 0\}$).
4.  Optionally rebalance by swapping boundary vertices to satisfy strict size constraints.

**Example 2 (Partitioning the Running Graph)**
The triangle $\{1,2,3\}$ in our example graph is tightly connected. A balanced cut that separates $\{1,2\}$ from $\{3,4,5\}$ would cut edges $(1,3)$ and $(2,3)$, resulting in a cut weight of 2. Another option, separating $\{1,2,3\}$ from $\{4,5\}$, cuts $(2,4)$ and $(3,5)$ (weight 2), but is unbalanced ($3$ vs $2$ nodes). The spectral method balances these trade-offs, often identifying the "bottleneck" in the graph.

## 4 Maximum Cut (Max-Cut)

![DSP96000](/personal_blog/VLSI_layout.jpg)
*(Caption includes link: [DSP96000](https://semiwiki.com/eda/7121-is-there-anything-in-vlsi-layout-other-than-pushing-polygons-3/))*

In contrast to graph partitioning, the Maximum Cut (Max-Cut) problem seeks to partition the vertices to *maximize* the weight of edges between the sets. This has applications in VLSI circuit design, where edges might represent signal interference. A cut corresponds to assigning components to two different layers or regions to maximize the separation of interfering pairs. It also appears in statistical physics as the problem of finding the ground state of an anti-ferromagnetic Ising model.

### 4.1 Problem Formulation

**Definition 3 (Maximum Cut)**
Find a bipartition $V=S\cup \overline{S}$ maximizing:
$$\mathrm{cut}(S,\overline{S})=\sum_{i\in S,\,j\in \overline{S}} w_{ij}.$$

Using spin variables, we have:
$$\mathrm{cut}(S,\overline{S})=\frac14 \sum_{(i,j)\in E} w_{ij}(1-s_is_j).$$
Maximizing this quantity is equivalent to minimizing the sum of the spin products:
$$\min_{s\in\{-1,1\}^n}\ \sum_{(i,j)\in E} w_{ij}s_is_j.$$

### 4.2 SDP Relaxation (Goemans--Williamson)

Max-Cut is famous for the Goemans--Williamson algorithm, which utilizes Semidefinite Programming (SDP). The key idea is a "lifting" procedure: we relax the scalar spins $s_i$ to unit vectors $v_i \in \mathbb{R}^n$. The term $s_i s_j$ becomes the inner product $v_i^\top v_j$. The SDP relaxation is:

$$
\max_{X\succeq 0}\quad \frac14\sum_{(i,j)\in E} w_{ij}(1-X_{ij}) \quad\text{s.t.}\quad X_{ii}=1\ \forall i,
$$

where $X_{ij} = v_i^\top v_j$ is a Gram matrix. This is a convex problem solvable in polynomial time. To recover a discrete solution, we use a random hyperplane to split the vectors.

**Algorithm 2: Goemans--Williamson SDP Rounding for Max-Cut**
1.  Solve SDP to obtain $X^\star\succeq 0$ with $X^\star_{ii}=1$.
2.  Compute a factorization $X^\star = V^\top V$ (columns correspond to vectors $v_i$).
3.  Sample a random vector $r\sim\mathcal{N}(0,I)$.
4.  Set $s_i=\mathrm{sign}(v_i^\top r)$ and return the cut induced by $s$.

**Theorem 1 (Approximation Guarantee)**
The Goemans--Williamson algorithm achieves an approximation ratio of at least $0.878$ for weighted Max-Cut.

**Example 3 (Max-Cut on the Running Graph)**
To maximize the cut on our example graph, one tends to separate vertices within the triangle (since all edges are present) and also split the path structure $(2,4)$--$(4,5)$--$(5,3)$. Unlike balanced partitioning, Max-Cut has no size constraint; very unbalanced cuts may be optimal if they capture more edge weight.

## 5 Minimum Vertex Cover

The Minimum Vertex Cover problem addresses coverage. Imagine a network of routers where we wish to place traffic monitors. We want to select the smallest number of routers such that every communication link (edge) is observed by at least one monitor endpoint. This is also relevant in fault detection and sensor placement.

![Minimum Vertex Cover](/personal_blog/MinVertexCover_small.png)

### 5.1 Problem Formulation

**Definition 4 (Vertex Cover)**
A set $C\subseteq V$ is a vertex cover if every edge has at least one endpoint in $C$:
$$\forall (i,j)\in E:\quad i\in C\ \text{or}\ j\in C.$$
The minimum vertex cover problem is to minimize $|C|$.

Let $x_i=1$ if $i\in C$, else $0$. This leads to a canonical Integer Linear Program (ILP):
$$\min_{x\in\{0,1\}^n}\ \sum_{i=1}^n x_i \quad\text{s.t.}\quad x_i+x_j\ge 1\ \ \forall (i,j)\in E.$$

### 5.2 Relaxations and Approximations

**LP Relaxation and Half-Integrality:**
If we relax the constraint $x_i \in \{0,1\}$ to $x_i \in [0,1]$, we obtain a Linear Program (LP). For vertex cover on general graphs, this LP has a property called *half-integrality*: there always exists an optimal LP solution where $x_i \in \{0, \frac{1}{2}, 1\}$.

**Approximation via Maximal Matching:**
A simple, robust strategy is to use a maximal matching. A matching is a set of edges without common vertices. If we take both endpoints of a maximal matching, we are guaranteed to cover all edges, and the size of this set is at most twice the optimum.

**Algorithm 3: 2-Approximation for Minimum Vertex Cover (Maximal Matching)**
1.  $C\leftarrow\emptyset$.
2.  Compute a maximal matching $M$ (greedily add edges while possible).
3.  **for** each $(u,v)\in M$
4.  $C\leftarrow C\cup\{u,v\}$.
5.  **end for**
6.  Return $C$.

### 5.3 QUBO Penalty Formulation

Vertex cover can also be modeled as an unconstrained quadratic problem by adding penalty terms for uncovered edges:

$$
\min_{x\in\{0,1\}^n}\ \sum_i x_i + \lambda\sum_{(i,j)\in E} \bigl(1-x_i-x_j\bigr)^2, \qquad \lambda\gg 1.
$$

If an edge is uncovered ($x_i=x_j=0$), the penalty term contributes $\lambda$ to the cost.

**Example 4 (Vertex Cover on the Running Graph)**
Consider our graph. Edge $(4,5)$ forces at least one of $\{4,5\}$ into the cover. The triangle edges force at least two vertices among $\{1,2,3\}$ into the cover. A valid cover of size 3 exists: $C=\{2,3,4\}$. It covers all edges: $(1,2)$ and $(2,4)$ by node 2; $(1,3)$ and $(3,5)$ by node 3; $(4,5)$ by node 4.

## 6 Maximum Independent Set (MIS)

In wireless networks, an edge represents interference between two transmitters. An independent set corresponds to a group of transmitters that can broadcast simultaneously on the same channel without collisions. The goal is to maximize this set to maximize network throughput.

### 6.1 Problem Formulation

**Definition 5 (Independent Set)**
A set $S\subseteq V$ is independent if no edge is contained inside $S$:
$$\forall (i,j)\in E:\quad \text{not both } i,j\in S.$$
The maximum independent set problem maximizes $|S|$, denoted by $\alpha(G)$.

The ILP formulation is:
$$\max_{x\in\{0,1\}^n}\ \sum_i x_i \quad\text{s.t.}\quad x_i+x_j\le 1\ \ \forall (i,j)\in E.$$

### 6.2 Relationship to Vertex Cover

There is a fundamental complementarity between MIS and Vertex Cover.

**Proposition 1**
A set $C$ is a vertex cover if and only if $V\setminus C$ is an independent set. Consequently,
$$\alpha(G) + \tau(G) = n,$$
where $\alpha(G)$ is the maximum independent set size and $\tau(G)$ is the minimum vertex cover size.

This relationship allows us to translate algorithms and bounds from one problem to the other.

### 6.3 Greedy Heuristics

While MIS is hard to approximate in the worst case, greedy heuristics based on node degree often perform well on sparse graphs. The minimum-degree heuristic iteratively selects the node with the fewest neighbors (least potential for conflict).

**Algorithm 4: Greedy Heuristic for MIS (Minimum-Degree Rule)**
1.  $S\leftarrow\emptyset$, $G' \leftarrow G$.
2.  **while** $V(G')\neq\emptyset$
3.  Choose $v\in V(G')$ with minimum degree in $G'$.
4.  $S\leftarrow S\cup\{v\}$.
5.  Remove $v$ and its neighbors from $G'$.
6.  **end while**
7.  Return $S$.

**Example 5 (MIS on the Running Graph)**
Vertices $\{1,4\}$ form an independent set (no edge between them). The set $\{1,4,5\}$ is *not* independent because $(4,5)\in E$. Using the complement property: since we found a vertex cover of size 3 ($\{2,3,4\}$), the complement $\{1,5\}$ is an independent set of size 2.

## 7 Maximum Clique

A clique is a subset of vertices where every pair is connected. In protein-protein interaction networks, a clique often indicates a protein complex—a group of proteins that bind together to perform a function. In social networks, they represent tightly knit communities.

### 7.1 Problem Formulation

**Definition 6 (Clique)**
A set $K\subseteq V$ is a clique if every pair is connected:
$$\forall i\neq j\in K:\ (i,j)\in E.$$
The maximum clique problem maximizes $|K|$, denoted $\omega(G)$.

A useful viewpoint is via the complement graph $\overline{G}=(V,\overline{E})$, where $(i,j)\in\overline{E}$ iff $(i,j)\notin E. A clique in $G$ corresponds exactly to an independent set in $\overline{G}$. Thus:
$$\omega(G) = \alpha(\overline{G}).$$

### 7.2 Motzkin--Straus and Continuous Characterization

The Motzkin--Straus theorem provides a classic continuous characterization of the clique number, linking it to optimization over the probability simplex $\Delta_n$.

**Theorem 2 (Motzkin--Straus)**
For an unweighted graph with adjacency matrix $A$:
$$\max_{x\in\Delta_n} x^\top A x \;=\; 1 - \frac{1}{\omega(G)}.$$

### 7.3 Exact Search: Bron--Kerbosch

The Bron--Kerbosch algorithm is a standard exact method for enumerating maximal cliques. It uses recursive backtracking with efficient pruning mechanisms.

**Algorithm 5: Bron--Kerbosch with Pivoting (Maximal Cliques)**
1.  **procedure** $BK(R,P,X)$
2.  **if** $P=\emptyset$ **and** $X=\emptyset$
3.  Report $R$ as a maximal clique.
4.  **end if**
5.  Choose pivot $u\in P\cup X$.
6.  **for** each $v\in P\setminus N(u)$
7.  $BK(R\cup\{v\},\ P\cap N(v),\ X\cap N(v))$.
8.  $P\leftarrow P\setminus\{v\}$; $X\leftarrow X\cup\{v\}$.
9.  **end for**

**Example 6 (Clique on the Running Graph)**
In our example, vertices $\{1,2,3\}$ form a clique of size 3 (a triangle). No clique of size 4 exists because vertex 4 is disconnected from 1 and 3, and vertex 5 is disconnected from 1 and 2.

## 8 Unified View of Relaxations

We can view the relaxations discussed in this chapter through a unified lens. The transition from discrete hardness to continuous tractability typically follows a pattern of replacing discrete sets with geometric spaces (spheres, simplices, or cones of positive semidefinite matrices).

### 8.1 SDP Lifting: The Central Technique

A central concept connecting these problems is *SDP lifting*. We replace the product of scalar binary variables $s_i s_j$ with a matrix variable $X_{ij}$.

**Definition 7 (SDP Lifting)**
For $s\in\{-1,1\}^n$, define $X=ss^\top$. Then $X$ satisfies:
$$X\succeq 0,\quad \mathrm{rank}(X)=1,\quad X_{ii}=1.$$
Dropping the non-convex rank constraint yields a tractable SDP relaxation.

This template captures many binary quadratic problems:
$$\min_{s\in\{-1,1\}^n}\ s^\top Q s \quad \Longrightarrow \quad \min_{X \succeq 0} \text{Tr}(QX)\ \ \text{s.t.}\ X_{ii}=1.$$

### 8.2 Randomized Rounding

Once a continuous relaxation (like SDP) is solved, we need a generic method to return to discrete variables. The random hyperplane technique is the standard approach.

**Algorithm 6: Generic Random Hyperplane Rounding**
1.  Solve SDP and obtain $X^\star\succeq 0$.
2.  Factorize $X^\star=V^\top V$ to obtain vectors $v_i$.
3.  Sample $r\sim\mathcal{N}(0,I)$.
4.  $s_i\leftarrow \mathrm{sign}(v_i^\top r)$ for all $i$.
5.  Return discrete assignment $s$.

### Take Aways

* **Graph Partitioning:** For large sparse graphs (like meshes), use spectral methods followed by local refinement (e.g., Kernighan--Lin).
* **Max-Cut:** Use the Goemans--Williamson SDP for high quality and guarantees; use local search (flip heuristics) for speed on massive graphs.
* **Vertex Cover:** Use ILP solvers for exact solutions on manageable instances; use the matching-based 2-approximation for scalability.
* **MIS / Clique:** Use branch-and-bound with strong bounds (like the Lovász theta function) for exact solutions; use greedy heuristics for very large graphs.
* **Energy/QUBO/Ising:** Use this formulation when you want a unified modeling interface, particularly when exploring hardware acceleration (quantum annealing) or embedding discrete problems into machine learning pipelines.

## 9 Solving CO-Problems using GNN: The Data-Driven Shift

> Traditional combinatorial optimization relies on exact algorithms or hand-crafted heuristics that solve each problem instance in isolation, often neglecting the shared structural patterns inherent in industrial datasets. This chapter introduces *Neural Combinatorial Optimization*, a paradigm shifting towards data-driven solvers powered by Graph Neural Networks (GNNs). We classify these neural approaches into three distinct categories.
>
> * First, *Primal approaches* directly predict solutions, utilizing supervised imitation of exact solvers, reinforcement learning for sequential construction, or physics-inspired unsupervised learning that relaxes discrete Hamiltonians (QUBO/Ising models) into differentiable loss functions.
> * Second, *Dual approaches* integrate GNNs into exact frameworks like Branch-and-Bound to accelerate critical decisions such as variable branching, maintaining optimality guarantees while reducing runtime.
> * Finally, we explore *Algorithmic Reasoning*, a method for designing neural architectures that structurally align with classical algorithms (e.g., dynamic programming), thereby significantly improving generalization to large-scale, out-of-distribution instances.

In the previous chapter, we explored classical approaches to Combinatorial Optimization (CO) on graphs. We discussed exact methods like branch-and-bound, which guarantee optimality but suffer from exponential time complexity, and approximation algorithms or heuristics, which sacrifice optimality for speed. However, a fundamental limitation of these traditional methods is that they view every problem instance in isolation. For example, a routing algorithm for a delivery company might solve thousands of instances of the Traveling Salesperson Problem (TSP) every day. These instances likely share common structural patterns—nodes might cluster in city centers, or traffic patterns might dictate similar edge weights. Classical algorithms start from scratch every time, ignoring this rich historical data. This realization has led to the emergence of Neural Combinatorial Optimization: the use of machine learning, and specifically Graph Neural Networks (GNNs), to solve or assist in solving hard optimization problems. The core premise is simple yet powerful: instead of hand-crafting heuristics, we can *learn* them from data. In this chapter, we will explore how GNNs act as a bridge between discrete optimization and differentiable deep learning. We will examine three distinct paradigms:

1.  Supervised Learning: Training GNNs to mimic optimal solutions found by exact solvers.
2.  Reinforcement Learning: Learning constructive heuristics by maximizing a reward (the objective function).
3.  Unsupervised / Physics-Inspired Learning: Encoding the optimization problem directly into a differentiable loss function, allowing the GNN to find low-energy states without any labeled data.

### 9.1 Graph Neural Networks: The Engine of Reasoning

Before applying them to optimization, we must understand why GNNs are the architecture of choice. CO problems are inherently relational; the "meaning" of a variable (node) is defined by its interactions (edges) with other variables. Standard neural networks (MLPs) or Convolutional Neural Networks (CNNs) are ill-suited for this because they expect fixed-size inputs or grid-like structures. Graphs are irregular and permutation invariant—renaming the nodes does not change the problem. A GNN operates by iteratively updating a vector representation (embedding) for every node in the graph. Let $h_v^{(k)}$ be the feature vector of node $v$ at layer $k$. The update rule, known as Message Passing, generally follows this form:

$$
h_v^{(k)} = \sigma \left( W_1^{(k)} h_v^{(k-1)} + W_2^{(k)} \sum_{u \in \mathcal{N}(v)} h_u^{(k-1)} \right)
$$

where:
* $\mathcal{N}(v)$ is the set of neighbors of node $v$.
* $W_1^{(k)}$ and $W_2^{(k)}$ are learnable weight matrices.
* $\sigma$ is a non-linear activation function (e.g., ReLU or Sigmoid).
* The sum operator $\sum$ aggregates messages from neighbors. It is crucial because it is permutation invariant: the order of neighbors does not matter.

Through $K$ layers of message passing, a node accumulates information from its $K$-hop neighborhood. In the context of optimization, this embedding $h_v^{(K)}$ encodes the node's role in the local graph structure, allowing the network to make informed decisions about whether to include the node in a set, cut an edge, or branch on a variable.

### 9.2 Primal Approaches: Finding Solutions Directly

Primal approaches use GNNs to directly output a feasible solution to the optimization problem. The GNN acts as a heuristic that maps a raw graph instance to a solution vector. The most straightforward approach is Imitation Learning. If we have a dataset of graphs and their optimal solutions (generated by a slow, exact solver like Gurobi), we can train a GNN to predict these solutions. For a problem like Minimum Vertex Cover, we treat it as a binary classification task. The GNN outputs a probability $p_v \in [0,1]$ for each node $v$, representing the likelihood that $v$ is in the optimal cover. We train using Binary Cross Entropy loss against the ground truth labels.

**Limitations:**
* Data Scarcity: Generating labeled data requires solving NP-hard problems to optimality, which is computationally expensive for large graphs.
* Generalization: A model trained on small, solvable graphs may fail to generalize to the massive graphs encountered in real-world applications.

### 9.3 Reinforcement Learning (RL) for Construction

To overcome the need for labeled data, we can use Reinforcement Learning. Here, the GNN acts as a policy $\pi(a|s)$ that constructs a solution step-by-step.

Consider the Traveling Salesperson Problem (TSP).
1.  State ($s_t$): The partial tour constructed so far and the graph structure.
2.  Action ($a_t$): Selecting the next city to visit from the set of unvisited cities.
3.  Reward: The negative length of the final tour (since we want to minimize distance).

The GNN encodes the graph, and a mechanism (often an attention mechanism or "Pointer Network") selects the next node based on the current embedding. The system learns by trial and error to maximize the reward. This approach, famously demonstrated in the "Attention, Learn to Solve Routing Problems!" papers, allows the model to learn heuristics that often outperform human-designed ones like Nearest Neighbor or Christofides, especially on random instances.

### 9.4 Physics-Inspired Unsupervised Learning

A powerful recent development is the Physics-Inspired approach. Instead of mimicking a solver (supervised) or learning from sparse rewards (RL), we can encode the mathematical definition of the problem directly into a differentiable loss function. This transforms the discrete combinatorial problem into a continuous optimization problem that can be solved via gradient descent. This framework is particularly effective for problems expressible as Quadratic Unconstrained Binary Optimization (QUBO) tasks, which maps directly to the Ising Model in physics.

**The QUBO Hamiltonian**
Recall that many graph problems can be written as minimizing a quadratic cost function of binary variables $x_i \in \{0, 1\}$:

$$H_{\text{QUBO}}(x) = \sum_{i,j} x_i Q_{ij} x_j$$

where $Q$ is a matrix defining the problem instance. For example, in the Maximum Cut (Max-Cut) problem, we want to maximize the number of edges between two sets. This is equivalent to minimizing the energy of an antiferromagnetic Ising model. The Hamiltonian is:

$$
H_{\text{MaxCut}} = \sum_{(i,j) \in \mathcal{E}} A_{ij} (2 x_i x_j - x_i - x_j)
$$

where $A_{ij}$ is the adjacency matrix. If nodes $i$ and $j$ have different values ($0$ and $1$), the term contributes to the cut. Standard backpropagation cannot optimize discrete binary variables. The physics-inspired strategy employs a relaxation:
1.  Continuous Relaxation: We relax the binary constraint $x_i \in \{0, 1\}$ to a continuous probability $p_i \in [0, 1]$.
2.  GNN Parameterization: A GNN accepts the graph structure (and random node initialization) and outputs a continuous value $p_i(\theta)$ for every node, where $\theta$ represents the network weights. A Sigmoid activation at the final layer ensures the output is in $[0, 1]$.
3.  Differentiable Loss: We substitute these probabilities directly into the QUBO Hamiltonian to define the loss function:

$$\mathcal{L}(\theta) = \sum_{i,j} p_i(\theta) Q_{ij} p_j(\theta)$$

This loss function measures the "expected energy" of the system. By minimizing $\mathcal{L}(\theta)$ via gradient descent (e.g., using Adam), the GNN learns node embeddings that push the system toward a low-energy state (a good solution). After training, the GNN outputs continuous values (e.g., $p_i = 0.9$ or $p_j = 0.1$). To get a valid discrete solution, we apply a simple projection:
$$x_i = \begin{cases} 1 & \text{if } p_i \ge 0.5 \\ 0 & \text{if } p_i < 0.5 \end{cases}$$

Crucially, this method is unsupervised. It requires no labeled training data. The GNN essentially acts as a "neural annealer," settling the system into a low-energy configuration.

**Example 7 (Solving Max-Cut with Physics-GNN)**
Consider a graph with 1 million nodes.
1.  We initialize the GNN with random node features.
2.  The GNN performs message passing to understand local topology.
3.  The output layer predicts a "spin" probability for each node.
4.  The loss function calculates the cut weight based on these probabilities (in a soft, differentiable manner).
5.  Gradients flow back to update the GNN weights, teaching it that connected nodes should have opposite probabilities (anti-correlated).
6.  Once converged, we threshold the outputs to get the partition.
This approach scales linearly with the number of edges, allowing it to solve million-variable problems that are impossible for traditional SDP solvers.

For problems with hard constraints, like the Maximum Independent Set (MIS) (no two selected nodes can be connected), we use penalty terms. The Hamiltonian combines a maximization term (size of the set) and a penalty term (independence violation):

$$H_{\text{MIS}} = - \sum_{i} x_i + P \sum_{(i,j) \in \mathcal{E}} x_i x_j$$

Here, the first term encourages $x_i=1$ (selecting nodes), while the second term adds a large penalty $P$ if two connected nodes are both selected ($x_i x_j = 1$). The GNN minimizes this combined objective, learning to balance the desire to pick nodes with the necessity of avoiding conflicts.

### 9.5 Dual Approaches: Enhancing Exact Solvers

While primal approaches replace the solver, dual approaches aim to improve the exact solvers themselves. Modern solvers like CPLEX or Gurobi rely on complex heuristics for decisions like "which variable should I branch on next?" (Branching) or "which node should I explore?" (Node Selection). Branch-and-Bound (B\&B) works by recursively splitting the search space. At each step, selecting the "right" variable to split can dramatically reduce the tree size. The standard heuristic, Strong Branching, is very effective but computationally expensive (it requires solving hypothetical LPs). We can train a GNN to imitate Strong Branching. 
* Input: A bipartite graph representing the current state of the solver (Variables on one side, Constraints on the other).
* Message Passing: Information flows between variables and the constraints they appear in. This captures the structure of the Linear Program.
* Output: A score for each variable predicting its quality as a branching candidate.

Once trained, the GNN is fast to evaluate. Inside the solver, we replace the slow calculation of Strong Branching with the fast inference of the GNN. This hybrid approach preserves the optimality guarantee of the exact solver while potentially speeding up the runtime by orders of magnitude.

### 9.6 Algorithmic Reasoning

An emerging frontier discussed in the literature is Algorithmic Reasoning. This paradigm seeks to build neural networks that align with the logical steps of classical algorithms (like Dynamic Programming or Breadth-First Search). The concept of Algorithmic Alignment suggests that a neural network can learn a task efficiently only if its architecture can represent the underlying algorithm's logic. For example, GNNs align well with dynamic programming algorithms like Bellman-Ford because the message-passing step ($h_v \leftarrow \sum h_u$) structurally resembles the relaxation step ($d_v \leftarrow \min (d_u + w_{uv})$). By training GNNs to execute these algorithmic steps, we can create "neural executors" that are robust to noise and can generalize to inputs far larger than those seen during training.

## 10 Conclusion and Future Outlook

The integration of Graph Neural Networks into Combinatorial Optimization represents a significant leap forward. We are moving from static, hand-designed algorithms to dynamic, data-driven systems.

| Paradigm | Mechanism |
| :--- | :--- |
| **Primal (Constructive)** | RL agents build solutions sequentially (e.g., TSP tours). Good for routing and scheduling. |
| **Primal (Physics-Inspired)** | Unsupervised GNNs minimize a differentiable Hamiltonian (QUBO). Excellent for large-scale spin glasses, Max-Cut, and MIS. |
| **Dual (Hybrid)** | GNNs predict decisions (branching/cutting) inside exact solvers. Maintains optimality guarantees while improving speed. |

**Future Directions:**
Current research focuses on scalability (handling graphs with billions of nodes), generalization (training on small graphs and solving large ones), and robustness. As hardware accelerators for deep learning improve, we can expect "Neural Solvers" to become a standard part of the industrial optimization toolkit, tackling problems in logistics, finance, and science that were previously out of reach.
