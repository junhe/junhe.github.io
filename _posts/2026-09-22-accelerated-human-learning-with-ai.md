---
layout: post
title: "Accelerated Human Learning with AI, for Engineers"
---

At Google, the idea of the Elephant and the Goldfish is popular. It recommends having an agent with a long-term memory of the overall architecture and plan, and many goldfish that start fresh and focus on small-scope tasks. This is not surprising. Human society has been doing this for a long time; for example, shop owners distribute tasks to apprentices, who focus on smaller tasks.

Using the same analogy, I think great tech leads are like elephants: they have long-term memory of the overall architecture and plans for one or multiple projects. They guide and distribute tasks to other team members who focus on smaller scopes.

Becoming a great tech lead is hard. There are a lot of docs and code to read, and a lot of experience to gain. Fortunately, we now have AI, which can retrieve information much faster than traditional search engines. What was not possible before AI is that AI can test your knowledge, which is a critical part of learning.


This article describes a structured way of learning in the era of AI. It should be applicable to learning any topic. The idea is simple: **manually write a paper by recursively reading and testing via AI**.

## Why write a paper manually?

A handwritten paper reflects your mind. If you can organize the paper well, it means that your mind is well-organized. In the meantime, by forcing yourself to organize the paper, you force yourself to understand the structure of the project.


It is easy to understand why we have to write the paper manually, instead of using AI. A lot of times, we ask AI something and read the answer. We think we understand it. But if you close the tab and ask the question again, you are often still clueless.


Start your paper with the following sections and gradually fill in the content manually:


1. Abstract  
2. Introduction  
3. Background  
4. Related Work  
5. Design  
6. Evaluation  
7. Conclusion  
8. References


Always keep the paper organized. If you have some messy notes, use a scratch doc to host them temporarily. Additionally, if you have content but don't know exactly where to put it, do your best to place it somewhere. Often, it becomes obvious where it should go after a while.


Make the paper flow well; this forces you to connect the knowledge you have acquired.

## Read and Test with AI

Essentially, we want to learn like Richard Feynman, who strongly believed that you should close your book and test your understanding after reading. If you fail, you should learn it again. This process forces knowledge into your brain and back out. The following diagram illustrates the process:


![image.png](/assets/images/accelerated-human-learning-with-ai/accelerated-learning.png){: width="500px"}


Nowadays, you can almost get information directly from AI. You can ask clarification questions. If you think the AI is hallucinating, you can press further, and it will usually produce the correct answer.


The testing part is interesting. There are many ways to validate whether you have the correct knowledge. You can "close the book" and write down notes; you can apply the knowledge by writing code; you can add logging to the system and run the code; you can try explaining the concept to others (including AI); or you can ask an AI to quiz you.

## Update the Paper

After validating your knowledge, update the paper. If you have already written notes during the testing step, you can simply move those notes into the paper. (You can also directly test your knowledge by writing or updating the paper.)


When updating the paper, think about where the content fits best and how the overall flow should be adjusted.

## Example

For example, if you just moved to a new team and need to understand a new project, this is what you might do:


1. Collect a bunch of design docs and code pointers, and add them to the References section.  
2. Start understanding why the project was started and fill out the Background section.  
3. When you encounter a similar project (e.g., from another team or company), update the Related Work section.  
4. Each time you read about a specific component of the system, add a new subsection to Design.  
5. …

Now, learning becomes "filling in the paper," which is very concrete. In the end, you will have structured artifacts that you can easily refer back to.
