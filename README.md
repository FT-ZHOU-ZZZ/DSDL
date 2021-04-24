# Deep Semantic Dictionary Learning for Multi-label Image Classification

The current project page provides [pytorch](http://pytorch.org/) code that implements the following paper:   
**Title:**      "Deep Semantic Dictionary Learning for Multi-label Image Classification"    
**Authors:**     Fengtao Zhou, Sheng Huang, Yun Xing    
**Project Page:**          

**Abstract:**  
Compared with single-label image classification, multi-label image classification is more practical and challenging. Some recent studies attempted to leverage the semantic information of categories for improving multi-label image classification performance. However, these semantic-based methods only take semantic information as type of complements for visual representation without further exploitation. In this paper, we present a innovative path towards the solution of the multi-label image classification which considers it as a dictionary learning task. A novel end-to-end model named Deep Semantic Dictionary Learning (DSDL) is designed. In DSDL, an auto-encoder is applied to generate the semantic dictionary from class-level semantics and then such dictionary is utilized for representing the visual features extracted by Convolutional Neural Network (CNN) with label embeddings. The DSDL provides a simple but elegant way to exploit and reconcile the label, semantic and visual spaces simultaneously via conducting the dictionary learning among them. Moreover, inspired by iterative optimization of traditional dictionary learning, we further devise a novel training strategy named Alternately Parameters Update Strategy (APUS) for optimizing DSDL, which alteratively optimizes the representation coefficients and the semantic dictionary in forward and backward propagation. Extensive experimental results on three popular benchmarks demonstrate that our method achieves promising performances in comparison with the state-of-the-arts.

## Requirements
Python = 3.5

PyTorch = 0.4.1

## Code for multi-label image classification on three benchmark datasets VOC 2007, VOC 2012, MS-COCO (will come up soon).
