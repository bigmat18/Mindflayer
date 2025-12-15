---
Data: 2025-12-15T18:53:00
Tags:
  - note
  - youngling
Connection:
  - "[[Scientific and Large Data Visualisation]]"
Area: "[[Master's degree]]"
---
# Channels

Visual channels are a way to control the appearance of marks and encode information about attributes independently of the dimensionality of marks. Main channels:
- **Spatial position**: aligned planar, unaligned planar, depth, region
- **Color**: hue, saturation, luminance
- **Shape and texture**
- **Slope and angle** (tilt)
- **Size**: length (1D), area (2D), volume (3D)

![[Pasted image 20251215182212.png | 400]]
#### Example
###### Bar charts
- Bar charts encode two attributes, a quantitative one (y-axis, height of the bars) and a categorical one (to spread the bars along the x-axis)
- **Mark**: lines (bars, 1D marks)
- **Channel 1**: vertical spatial position for the quantitative attribute
- **Channel 2**: horizontal spatial position for the categorical attribute

![[Pasted image 20251215182344.png | 200]]

###### Scatterplot (V1)
V1: Encoding two quantitative attributes
- **Marks**: points (circles, 0D marks)
- **Channel 1**: vertical spatial position (along y-axis) for the first quantitative attribute 
- **Channel 2**: horizontal spatial position (along x-axis) for the second quantitative attribute

![[Pasted image 20251215182621.png | 200]]
###### Scatterplot (V2)
V2: Encoding three attributes
- **Marks**: points (circles, 0D marks)
- **Channel 1**: vertical spatial position (along y-axis) for the first quantitative attribute
- **Channel 2**: horizontal spatial position (along x-axis) for the second quantitative attribute
- **Channel 3**: color for the third attribute (either quantitative or qualitative)

![[Pasted image 20251215182506.png | 200]]
###### Scatterplot (V3)
In this last case we add a different area (2D) to the fourth quantitative attribute (channel 4).
- **Marks**: points (circles, 0D marks)
- **Channel 1**: vertical spatial position (along y-axis) for the first quantitative attribute
- **Channel 2**: horizontal spatial position (along x-axis) for the second quantitative attribute
- **Channel 3**: color for the third attribute (either quantitative or qualitative)
- **Channel 4**: area (2D) for the fourth quantitative attribute

![[Pasted image 20251215182721.png | 200]]

##### Contextual Components
Contextual components are elements that make it easier to interpret a visualization
- Legends, labels, annotations
- Grids, axes, reference lines

![[Pasted image 20251215182825.png | 200]]
##### Visual decoding
Visual decoding means deconstructing a visual representation it into its major units, and identifying:
- the graphical elements
- what are the visual marks? What are the visual channels?
- the mapping rules (i.e., the information that the graphical elements represent)
- what data items do the mark represent? What attributes do the channel represent?
It is useful for evaluating and redesigning visualizations

![[Pasted image 20251215182924.png | 400]]

- Identify marks, and the data items they stand for: e.g points (dots, circles), standing for medals
- Identify channels, and the attributes they stand for

![[Pasted image 20251215183058.png | 450]]

In this example:
- x-position, standing for distance from Bolt
- y-position, standing for time
- color, standing for gold, silver and bronze

### Channel types
Channel can be divided into two different categories, according to the two different sensor modalities oh the human perceptual system: **identity** and **magnitude** channels
- **Identity**: Identity channels give information about what something is or where it is. e.g., shape, color
- **Magnitude**: Magnitude channels tell us how much of something there is. e.g., length, area, volume, luminance and saturation, size, angle

All channels are not equal: the same data attribute encoded with two different visual channels will result in different information content on the user’s side, after it has passed through the perceptual and cognitive pathways of the human visual system

The use of marks and channels in vis design should be guided by the principles of **expressiveness** and **effectiveness**:
- There is a ranking of channels according to the type of data that is being visually encoded
- Once you have identified the most important pieces of information in your data, you have to ensure they are encoded with the highest-ranked channels
###### Expressiveness
The visual encoding should express all of, and only, the information that is present in the dataset attributes. Do not represent information and relationships that are not in the data:

- Do not use color (or position) when it does not encode any information
![[Pasted image 20251215184058.png | 200]]

- Ordered data should be shown so that our visual system perceives them as ordered, use magnitude channels
![[Pasted image 20251215184119.png | 250]]

- Conversely, unordered data should not be shown in a way that perceptually implies an ordering that does not exist
![[Pasted image 20251215184148.png | 450]]
###### Effectiveness
The importance of the attribute should match the salience of the channel, i.e., its noticeability:
- relevant information should be prioritized then encoded with the most effective/accurate channels to be most noticeable
- decreasingly important attributes can be matched with less effective channels

#### Channels Ranking
How effective are channels at conveying different types of attributes? One of the (possible) summaries for quantitative attributes (lots of research on this)

![[Pasted image 20251215184548.png | 400]]

A general table:
![[Pasted image 20251215184615.png | 300]] ![[Pasted image 20251215184815.png | 350]]

- Both lists have channels related to spatial position at the top
- The spatial channels are the only ones appearing in both lists
- The choice of which attributes to encode with position is the most central choice in visual encoding, as they will dominate the user’s mental model (i.e., internal representation used for reasoning)
- Depth/3D is low ranked
- While one can use a magnitude channel for categorical data or an identity channel for ordered data, that would be a poor choice because it violates the expressiveness principle

###### Example
![[Pasted image 20251215184938.png | 500]]

- Aim: comparing the market capitalization of 15 major banks before and after the banking crisis (compare small green circles to the larger blue circles to see how much the market capitalization declined)
- Drawback: Human visual perception does not support the accurate comparison of 2-D areas
- Also: do the differences in the numbers reflect the differences in the areas?

![[Pasted image 20251215185022.png | 450]]

Possible answers: accuracy, discriminability, separability, ability to provide visual popout, ability to provide perceptual groupings
### [[Accuracy]]
### [[Discriminability]]

### [[Separability]]

### [[Popout]]

### [[Relative vs Absolute Judgements]]
# References