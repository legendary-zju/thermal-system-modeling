<img width="709" height="512" alt="image" src="https://github.com/user-attachments/assets/adf6d6f4-ecd0-4a8c-a944-66e31902d97f" />

The thermal model in my research is in GasSteamCombineCyclePlantModel1.py, which is a gas-steam combine cycle.
The fundamental fluid-property solving-equation (like, h(p, T)) is invoked from Tespy, saving time for constructing fluid solving engine.
Thanks for open source library Tespy, the framework of which has given some reference for us.
Our research comes from the limitation of the application of Tespy.
To solve these problems in thermal systems, we spent a year repeatedly trying and finally found a successful path.
