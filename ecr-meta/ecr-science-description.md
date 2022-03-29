# Background
Flooding of rivers, streams, or other water reservoirs can cause damages and safety risks to people. To monitor the water level of those resources, we collected images from our nodes and analyze them using computer vision methods.

# Algorithm Description
As a method to preprocessing images to automatically draw line showing water level, we utilized top hat filtering and morphological closing method. The application estimates water level from a preprocessed using pre-defined water level map.

# Using the code
Output: water level (cm)  
Input: single image  
Image resolution: 960x1296  
Inference (calculation) time:  
Model loading time: N/A  

# Arguments
'-threshold': water level determination threshold  
'-roi-coordinates': X,Y Coordinates of region of interest for perspective transform
                    (default="448,280 500,800 520,800 470,280")  
'-new-coordinates': X,Y Coordinates of new region of interest for perspective transform
                    (default="0,0 0,600 100,600 100,0")  
'-pallet': X,Y Length of new pallet for perspective transform
                    (default="100,780")  

# Ontology:
The code publishes measurements with toptic ‘env.water.level’

# Inference from Sage codes
To query the output from the plugin, you can do with python library 'sage_data_client':
```
import sage_data_client

# query and load data into pandas data frame
df = sage_data_client.query(
    start="-1h",
    filter={
        "name": "env.water.level",
    }
)

# print results in data frame
print(df)
```
For more information, please see [Access and use data documentation](https://docs.sagecontinuum.org/docs/tutorials/accessing-data) and [sage_data_client](https://pypi.org/project/sage-data-client/).

### Reference
[1] Sabbatini, Luisiana, Lorenzo Palma, Alberto Belli, Francesca Sini, and Paola Pierleoni. "A Computer Vision System for Staff Gauge in River Flood Monitoring." Inventions 6, no. 4 (2021): 79.
