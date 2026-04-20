# State of the Art: WiFi CSI Human Pose Estimation

## 1. The Core Challenge: The Flailing Problem
Estimating a 3D or 2D human skeleton from pure Radio Frequency (RF) telemetry—specifically Channel State Information (CSI)—is profoundly different from computer vision. 
In computer vision, pixels have structural continuity; an elbow is visually connected to a wrist. In RF space, the subcarrier phase/amplitude matrices represent spatial disturbances, **but they do not inherently contain structural information.** 

When using a standard Neural Network (like our current Transformer + Linear Regressor pipeline) to predict `(x, y)` coordinates, the network treats every single joint as an independent variable. As observed in our `dualhead_straitjacket` runs, the loss plateaus because fixing the position of one joint (e.g., wrist) breaks the distance/angle to its parent (e.g., elbow). The model fails to understand that these two points are physically anchored together.

## 2. Current State of the Art (SOTA) Approaches
Modern academic research models have successfully solved this by completely abandoning flat coordinate regression and introducing **Topology Restraints**. 

### A. Graph Convolutional Networks (GCNs)
The most successful modern architecture for CSI Pose Estimation is the integration of **Graph Convolutional Networks (e.g., GraphPose-Fi)**.
* **How it Works**: A human skeleton is a mathematical **Graph**. Joints are *Nodes*, and Bones are *Edges*. 
* **The Architecture**: 
  1. A CNN or Transformer extracts pure features from the CSI tensor (identifying location and momentum).
  2. The output is fed, not into a linear layer, but into a GCN block.
  3. The GCN explicitly shares weights across the rigid skeletal edges (the 17 YOLOv8 keypoints). The model literally cannot move the "wrist" node without the math propagating directly up the "lower arm" edge to move the "elbow" node. 
* **Result**: Perfectly rigid, human-shaped skeletons that never spaghetti.

### B. WiPose: Forward Kinematics Integration (CNN + LSTM)
* **How it works**: WiPose achieved early SOTA by hard-coding prior knowledge of human skeletal structures physically into the loss and decoding mechanisms. 
* **Input Features**: It utilizes not just amplitude, but 3D velocity profiles extracted from the phase variance, allowing the system to distinguish posture-related features from random room multipath noise.

## 3. Recommended Path Forward
The pure Transformer architecture we have now is essentially a high-end implementation of the *first half* of a SOTA paper. Our model (the CNN spatial embedding + Transformer Temporal encoding) accurately extracts the latent RF features, but we drop the ball at the absolute very end by using `nn.Linear(256, 34)`.

**To fix SaferPINN, we must upgrade the `Biomechanic Head` into a Graph Convolutional Network (GCN).** 
By defining the Adjacency Matrix of our YOLOv8 17-point skeleton, we will physically force the output coordinates to move rigidly as a biological unit.
