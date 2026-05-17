# Thermal System Simulation
(be updating)
## Catalog:
<!-- TOC -->
-[abstract](#)   
├─ 📂Aurora   ------------------------------------------------------------ program package   
│   ├─ 📂components  
│   │   ├─ 📂electric components  
│   │   │   ├─ 🐍electric component   
│   │   │   ├─    
│   │   │   └─    
│   │   ├─ 📂fluid components   
│   │   │   ├─ 🐍fluid component    
│   │   │   ├─ 📂heat exchangers   
│   │   │   │   ├─ 🐍heat exchangeer    
│   │   │   │   ├─ 🐍evaporator    
│   │   │   │   ├─ 🐍over heater    
│   │   │   │   ├─ 🐍condenser    
│   │   │   │   ├─ 🐍extract heat exchanger    
│   │   │   │   ├─ 🐍boiler simple    
│   │   │   │   └─ 🐍evaporator    
│   │   │   ├─ 📂deaerators   
│   │   │   └─ 📂turbomachinery   
│   │   └─   
│   ├─ 📂combines  
│   ├─ 📂controls  
│   │   └─ 🐍controller   
│   ├─ 📂connections  
│   │   ├─ 🐍connection   
│   │   ├─ 🐍electric connection   
│   │   ├─ 🐍fluid connection   
│   │   └─ 🐍bus   
│   ├─ 📂nodes  
│   │   ├─ 🐍node   
│   │   ├─ 🐍electric node   
│   │   └─ 🐍fluid node   
│   ├─ 📂networks  
│   │   ├─ 🐍network   
│   │   └─    
│   │  
│   │  
├─ 📂master model    ----------------------------------------------------- model   
│   ├─ 📂GasSteamCombinePlant  
│   │   ├─ 🐍combine plant model   
│   │   ├─ 📄log   
│   │   ├─ 📊design data   
│   │   ├─ 📊offdesign data  
│   │   └─ 🖼image   
│   ├─ 📂PureSteamPlant  
│   │   ├─ 🐍pure steam plant model   
│   │   ├─ 📄log   
│   │   ├─ 📊design data   
│   │   ├─ 📊offdesign data  
│   │   └─ 🖼image   
│   └─ 📂

<!-- TOC -->



## Examples:
1. GasSteamCombinePlant
<img src="master_model/combine_plant/constructure_image/combine_plant_constructure.jpg" width="600">
The thermal model in my paper is in GasSteamCombineCyclePlantModel1.py at combine_plant of master_model, which is a gas-steam combine cycle.
The properties set are as follows:
<img src="master_model/combine_plant/constructure_image/data.png" width="300">
The relative bias of design condition with commercial software is as follows:
<img src="master_model/combine_plant/constructure_image/relative_bias_of_design.jpg" width="300">
2. PureSteamPlant
<img src="master_model/pure_steam_plant/constructure_image/pure_steam_plant_constructure.jpg" width="600">
Another model is PureSteamPlant, which has complex topology constructure. The energy comes from the BoilerSimple. 
The relative bias of design condition with commercial software is as follows: 
<img src="master_model/pure_steam_plant/constructure_image/relative_bias_of_design.jpg" width="300">
3. SolarThermalCombinePlant
<img src="master_model/solar_thermal_hybrid_plant/constructure_image/solar_thermal_hybrid_plant_constructure_image.jpg" width="600">



## Explanation:
The fundamental fluid-property solving-equation (like, h(p, T)) is invoked from Tespy, saving time for constructing fluid solving engine.
Thanks for open source library Tespy, the framework of which has given some reference for us.
Our research comes from the limitation of the application of Tespy.
To solve these problems in thermal systems, we spent a year repeatedly trying and finally found a successful path.
