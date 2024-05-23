Streamlit_0                  == Increment question's level App with already user-frienly features such as avatars, buttons, slider, etc.
Streamlit_1                  == Increment question's level doesn't increment by 1 level each time but adapts to user's capabilities. It also avoids to produce a question for a
                                C2 level user, preventing level index overflow. 
Streamlit_2(wSimCheck)       == Includes cosine similarity check between question and user answer and assesses a coherence score to avoid evaluating uncoherent answers
                                (e.g., user did not understand the question at all) or answers missing words.  
Streamlit_3(wSim&NatCheck)   == Asks for user's country of origin to determine if Native French and avoid proficiency check.
Streamlit_4(LocalTest)       == Tests the app locally w/o explicit openAI API key (taken from .env file through import os).
Streamlit_5(DeployTest)      == First version tested for the deploy, reading secret openAI API key from the Secrets section of App Management and downloading and unzipping model
                              folder from Google Drive as it cannot remain local or be uploaded on GitHub repo.
Streamlit_6(RefinedTest)     == Second version for the deploy test as suggested by chatGPT.
Streamlit_7(TestwErrLogging) == Third version tested for deploy with error logging as suggested by chatGPT. Tried to provide Cargo Rust Compiler through packages_Original.txt
                                and packages.txt. The first one introduces Cargo but it provides the old version to Streamlit. The other one doesn't provide Cargo as Rust 
                                Compilter was attempted to be provided within .py file.                                  
Streamlit_8(wRecommenderSystem) == As per Streamlit_3 with first attempt for recommender system of French texts (to be continued). 
