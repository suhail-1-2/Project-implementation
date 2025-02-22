import streamlit as st

def main():
    st.title("Smart Farming App")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Crop Recommendation", use_container_width=True):
            st.switch_page("pages/streamlit_crop_recommendation.py")  #streamlit_crop_recommendation.py
        if st.button("Crop and Fertilizer Prediction", use_container_width=True): 
            st.switch_page("pages/streamlit_crop_fertilizer.py")  #crop_andFertstreamlit.py
    
    with col2:
        if st.button("Fertilizer Prediction", use_container_width=True):
            st.switch_page("pages\streamit_fertilizer_prediction.py") #fert_pred.py
        # if st.button("LLM Chatbot", use_container_width=True):
        #     st.switch_page("llm_chatbot.py")

if __name__ == "__main__":
    main()
