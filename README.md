# Long-Term Effects of Revealing Ethnicity of Restaurant Owners on Consumer Behavior

This code is developed to explore the potential long-term impact on consumer behavior when the ethnicity of restaurant owners is disclosed. It is inspired by the findings in the paper "The Benefits of Revealing Race: Evidence from Minority-owned Local Businesses" by Abhay Aneja, Michael Luca, and Oren Reshef, which assesses the influence of such disclosures on consumer spending. The limitations in the temporal scope of the paper prevent it from providing definitive conclusions on the policy's effectiveness.

## Theoretical Framework

The approach taken by this code is to integrate the idea of revealing the ethnicity of restaurant owners with Thomas Schelling’s Segregation model. The foundational code is based on the methodologies described in "Segregation dynamics with reinforcement learning and agent-based modeling" by Egemen Sert, Yaneer Bar‐Yam, and Alfredo J. Morales.

## Implementation Details

- **Model:** Agent-Based Model (ABM)
- **Learning Technique:** Reinforcement Learning
- **Key Parameter:** `alpha`
  - **Description:** Measures the tolerance of individuals
  - **Range:** 0 to 1

## Usage

This simulation unfolds in two distinct episodes. In the initial phase, agents operate under the veil of ignorance regarding the ethnicity of restaurant owners, symbolized by a triangle on the result maps. Subsequently, in the second episode, agents are enlightened about the ethnic makeup. The distribution of restaurant ethnicities is predetermined, deliberately ensuring that Type B establishments constitute a minority. The simulation can iterate for any desired number of cycles.

Furthermore, the distribution of agents is exogenously arranged, guaranteeing a minority presence of Type B agents. Restaurant consumption is assessed using a point-based system; each visit by individuals from different ethnicities incrementally enhances the restaurant's score by 0.1 points.

Agents respond to a reward structure contingent on their tolerance threshold (`alpha`). For instance, if `alpha` is set at 1, patronizing a restaurant of a different ethnicity would yield a negative reward, computed as follows: `0.7 * (0.5 - alpha)`, resulting in -0.35. Conversely, dining at an establishment aligned with one's own ethnicity would yield a reward of 0.35. Additional rewards for movement and neighborhood interactions are incorporated to mimic the dynamics of a Schelling segregation model, aiming to explore the potential impact of restaurant consumption on agent segregation.

Adjustments to reward values can be made in the "mind.py" file, while the simulation itself runs in the "simulationABMRL" file. It's important to note that the `alpha` value should be modified across the "mind", "environment", and "simulationABMRL" files to ensure consistency.

This endeavor is strictly personal and devoid of any academic agenda, serving merely as a personal exploration prompted by questions raised during the reading of the paper authored by Abhay Aneja, Michael Luca, and Oren Reshef.


## Citation

If you utilize this code for academic purposes, please acknowledge the contribution by citing:

- **Last Name:** Maury
- **First Name:** Octavien
