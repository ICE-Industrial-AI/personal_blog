
---
title: "ML Techniques in Defensive and Offensive Cybersecurity"
description: "The integration of Artificial Intelligence into cybersecurity has created a dual-use dynamic. It is a tool of immense power for defenders (Blue Teams) seeking to automate threat detection, yet it simultaneously empowers attackers (Red Teams) to generate sophisticated, polymorphic threats. This article explores the spectrum of machine learning techniques—from standard supervised learning to cutting-edge Generative AI—and how they are applied on both sides of the digital battlefield."
pubDate: "Nov 30 2025"
heroImage: "/personal_blog/logo_CAS_AICybersecurity.png"
badge: "Latest"
---

*CAS AI-driven Cybersecurity and Strategic Defense*
*Eastern Switzerland University of Applied Sciences OST*  
# Machine Learning Techniques in Defensive and Offensive Cybersecurity
## From Supervised Learning to Multi-Agent RL - A Cybercirme Comic (Part I)

**Author:** *Christoph Würsch, Institute for Computational Engineering ICE, OST*  
**Date:** 30.11.2025


[Download the Presentation](/personal_blog/AI4Cybersecurity_(AI4Cyber)_WUCH.pdf) 

>The integration of Artificial Intelligence into cybersecurity has created a "dual-use" dynamic. It is a tool of immense power for defenders (Blue Teams) seeking to automate threat detection, yet it simultaneously empowers attackers (Red Teams) to generate sophisticated, polymorphic threats. This article explores the spectrum of machine learning techniques—from standard supervised learning to cutting-edge Generative AI—and how they are applied on both sides of the digital battlefield.

![Dual Use Nature of AI](/personal_blog/DualUse_blue_red_HD.png)



## 1. Introduction to AI in Cybersecurity

Artificial Intelligence is a double-edged sword in the cyber domain. The same algorithms used to detect patterns in network traffic can be used to obfuscate malware behavior.

### The Dual-Use Nature
* **Defensive AI (Blue Team):** Focuses on automated threat detection, pattern recognition at scale, and predictive analytics to prioritize vulnerability patching before exploits occur.
* **Offensive AI (Red Team):** Utilizes AI for automated fuzzing, exploit generation, evasion of traditional antivirus (AV), and creating hyper-personalized phishing campaigns via deepfakes and LLMs.

### Why AI is Critical: The Numbers
Traditional signature-based detection is failing under the sheer volume of modern threats.
* **Volume:** AV-TEST registers over **450,000** new malicious programs every single day.
* **Cost:** The global average cost of a data breach has risen to **$4.45 million**.
* **Speed:** AI-driven security automation is proven to reduce the breach lifecycle by an average of **108 days**.
* **Complexity:** Modern threats like "Fileless" malware and "Living off the Land" attacks require behavioral analysis, as there are no files to scan.

### Real-World Case Studies
To understand the stakes, we can look at specific implementations:
1.  **Offensive (DeepLocker):** A proof-of-concept by IBM Research. This "stealthy" malware hides its payload inside a benign video conferencing app. It utilizes facial recognition to execute the attack *only* when it recognizes a specific target's face via the webcam, remaining dormant and undetectable otherwise.
2.  **Defensive (Darktrace):** Known as the "Enterprise Immune System," this tool uses unsupervised learning to learn the "pattern of life" for every device. It famously detected the "WannaCry" ransomware in real-time by noticing anomalous SMB traffic, stopping the encryption before it spread.



## 2. Supervised Learning

**Definition:** Supervised learning involves training a model on a labeled dataset (Input $\rightarrow$ Output).  
**Primary Use:** Detecting known threat patterns where ground truth exists.

### Classification: Malware & Phishing
Classification algorithms predict a category (e.g., Malicious vs. Benign).

#### Example 1: Malware Detection
Defenders use **Deep Neural Networks (CNNs)** or **XGBoost**. A novel approach involves treating the raw bytes of a file as pixels in an image. CNNs can then detect visual structural patterns in the code that traditional obfuscation techniques fail to hide.

* **Offensive Counter-Measure:** Attackers use "Model Evasion." They train local proxy models to test if their malware is detected, modifying it iteratively until it bypasses the classifier.

> **Dataset:** [Microsoft Malware Prediction (Kaggle)](https://www.kaggle.com/c/microsoft-malware-prediction)

![AI Malware Detection](/personal_blog/AImalwareDetection_HD.png)

#### Example 2: Phishing & Spam Detection
Using **Natural Language Processing (NLP)** with models like BERT, defenders analyze email headers and semantic context (urgency, financial requests) to classify emails. Conversely, attackers use this logic to train models on a target's social media history to generate Spear-Phishing emails that mimic a colleague's writing style.

> **Dataset:** [Enron Email Dataset (Kaggle)](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)

### Regression: Scoring & Forecasting
Regression predicts continuous values rather than categories.

#### Example 1: Vulnerability Scoring
Using Linear Regression or Random Forests, systems can predict the **Common Vulnerability Scoring System (CVSS)** score (0.0 to 10.0) of a vulnerability before official analysis is released. This allows teams to prioritize patches immediately.

> **Dataset:** [NIST NVD Data Feeds](https://nvd.nist.gov/vuln/data-feeds)

#### Example 2: Network Traffic Forecasting
**Long Short-Term Memory (LSTM)** networks predict expected traffic volume. If actual traffic exceeds the prediction plus a threshold, a DDoS alert is triggered. Attackers use similar regression to determine the exact traffic spike size required to degrade service without triggering volume-based firewalls.

> **Dataset:** [CIC-IDS2017 (UNB)](https://www.unb.ca/cic/datasets/ids-2017.html)



## 3. Unsupervised Learning

**Definition:** Finding hidden patterns in unlabeled data.  
**Primary Use:** Zero-Day detection and Insider Threat detection. The system does not know what an attack looks like; it only knows what "normal" looks like.

### Anomaly Detection
Techniques like **Isolation Forests** or **Autoencoders** learn a baseline of normal traffic. Isolation Forests work by randomly selecting features and splitting values; anomalies are easier to isolate (require fewer splits) than normal data points. This is crucial for detecting data exfiltration at unusual hours.

> **Dataset:** [NSL-KDD Dataset (Kaggle)](https://www.kaggle.com/datasets/hassan06/nslkdd)

![Anomaly Detection](/personal_blog/AnomalyDetection_HD.png)

### User Behavior Analytics (UEBA)
Using **K-Means Clustering** or **DBSCAN**, users are grouped based on behavior (e.g., HR vs. IT department). If a user typically in the "Marketing Cluster" accesses a server usually only touched by the "SysAdmin Cluster," the system flags a compromised credential or insider threat.

> **Dataset:** [CERT Insider Threat Dataset (CMU)](https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099)

![User Behavior Analytics](/personal_blog/UserBehaviorAnalytics_HD.png)



## 4. Semi-Supervised Learning

### Password Cracking (GANs)
**Generative Adversarial Networks (GANs)**, such as PassGAN, revolutionize password cracking.
* **Generator:** Creates fake passwords.
* **Discriminator:** Tries to distinguish real human passwords (from leaks) from AI-generated ones.
The Generator learns the underlying probability distribution of human password choices, generating guesses that are far more effective than random brute force.

> **Dataset:** [RockYou.txt (SecLists)](https://github.com/danielmiessler/SecLists/blob/master/Passwords/Leaked-Databases/rockyou.txt.tar.gz)

### Malicious Domain Labeling
Using **Graph-based Label Propagation**, defenders can identify bad domains even with limited labeled data. By constructing a graph where nodes are domains and edges are shared attributes (same IP, registrar, or DNS), "Malicious" labels can be propagated to unknown nodes that are heavily connected to known bad actors.

> **Dataset:** [Malicious URLs Dataset (Kaggle)](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset)



## 5. Reinforcement Learning (RL)

### Automated Penetration Testing
Using **Deep Q-Networks (DQN)**, automated "Red Agents" learn to compromise systems.
* **Action Space:** Scan Port, Run Exploit, Move Laterally.
* **Rewards:** +100 for compromising a database, -10 for being detected.
The agent's goal is to find the optimal path to the "Crown Jewels".

> **Simulation Env:** [CyberBattleSim (Microsoft)](https://github.com/microsoft/cyberbattlesim)

![Automated Penetration Testing](/personal_blog/AutomatedPenetrationTesting_HD.png)

### Multi-Agent Honeypot Allocation
In a **Cooperative MARL** setting, defensive "Blue Agents" communicate to dynamically move high-interaction honeypots. They observe local traffic and coordinate to maximize the probability of trapping an active attacker.

> **Dataset:** [Honeypot Attack Logs (Kaggle)](https://www.kaggle.com/datasets/lougarou/honeypot-attack-logs)

![Honeypot Allocation](/personal_blog/HoneyPot_Allocation_HD.png)



## 6. Generative AI & LLMs

Generative AI represents the new frontier. Foundation models like GPT-4 or Llama 3 can generate code, text, and scripts, enabling hyper-personalized phishing and polymorphic malware.

### Offensive GenAI Case Studies

#### 1. Automated Social Engineering
Attackers scrape a target's LinkedIn or Twitter profile. An LLM then generates a context-aware email (e.g., referencing a specific recent conference) indistinguishable from human correspondence. "Dark LLMs" like WormGPT lack ethical guardrails, facilitating this at scale.

![Automated Social Engineering](/personal_blog/AutomatedSocialEngineering_HD.png)

#### 2. Malware Polymorphism (The Xillen Stealer)
**Polymorphism** is the ability of code to constantly mutate its identifiable features (signature) while maintaining its malicious functionality.

**The Xillen Stealer (v4-v5)** is a prime example of this evolution:
* **Concept:** The malware uses a polymorphic engine to ensure that every time it is downloaded, it has a unique file hash, rendering signature-based AV useless.
* **Mechanism:** It employs encryption loops (generating random keys per infection), junk code insertion (NOPs), and instruction substitution.
* **Python Specifics:** Being Python-based allows for dynamic obfuscation (scrambling variable names) and rapid iteration. The payloads are often "packed" (e.g., via PyInstaller), creating large, messy files that complicate analysis.

**AI Evasion Techniques:**
Xillen includes an `AIEvasionEngine` designed to defeat behavioral classifiers:
* **Noise Injection:** Performing random memory and CPU operations to confuse ML models.
* **Entropy Variance:** Altering the randomness of the file structure to avoid detection by entropy-based scanners.
* **Behavioral Mimicking:** It simulates mouse movements and waits for browser activity to appear human.

![Polymorphism Visualization](/personal_blog/Polymorphism_HD.png)

<img src="/personal_blog/XillenStealer_AI_target_detection.png" height="300">

*Xillen Stealer AI Target Detection*

<br>

<img src="/personal_blog/XillenStealer_AI_evasion_entropy.png" height="300">

*Xillen Stealer AI Evasion*

<br>

### Defensive GenAI Case Studies

#### 1. Automated Vulnerability Repair (APR)
LLM-driven patching goes beyond detection. It takes vulnerable code and an error log, and the LLM proposes a secure code candidate—for example, automatically rewriting raw SQL queries into parameterized queries to prevent injection.

#### 2. Threat Intel Summarization
LLMs digest thousands of unstructured reports (blogs, dark web feeds) and extract structured data (Actor, Target, IP) into JSON formats that can be immediately ingested by firewalls.



## Summary of Techniques & Datasets

| Category | Technique | Application | Dataset / Benchmark |
| - | - | - | - |
| **Supervised** | CNN / XGBoost | Malware Detection | [Microsoft Malware Prediction](https://www.kaggle.com/c/microsoft-malware-prediction) |
| **Supervised** | LSTM (RNN) | Traffic Forecasting | [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) |
| **Supervised** | NLP / SVM | Phishing/Spam | [Enron Email Corpus](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset) |
| **Unsupervised** | Isolation Forest | Anomaly Detection | [NSL-KDD](https://www.kaggle.com/datasets/hassan06/nslkdd) |
| **Unsupervised** | Clustering | User Behavior (UEBA) | [CERT Insider Threat](https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099) |
| **Semi-Supervised** | GANs | Password Cracking | [RockYou.txt](https://github.com/danielmiessler/SecLists) |
| **Semi-Supervised** | Label Prop. | Malicious URLs | [Malicious URLs Dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset) |
| **RL / MARL** | DQN | Auto Pen-Testing | [CyberBattleSim (Env)](https://github.com/microsoft/cyberbattlesim) |
| **GenAI (Offense)** | Dark LLMs | Auto-Phishing | [Human vs. LLM Phishing](https://www.kaggle.com/datasets/francescogreco97/human-llm-generated-phishing-legitimate-emails) |
| **GenAI (Defense)** | APR (LLM) | Auto-Patching | [LLM Vuln. Detection](https://zenodo.org/records/15108502) |

<br>

> This article highlights the rapid evolution of the cyber arms race. As defensive AI improves, offensive AI adapts with polymorphism and noise injection, necessitating continuous innovation in strategic defense.
