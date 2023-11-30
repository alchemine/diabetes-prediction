"""Tab page for streamlit app
"""

from diabetes_prediction.utils.app.io import *
from diabetes_prediction.utils.data.meta_dataloader import *


def show_generate_meta():
    def on_click_generate_meta_button():
        st.session_state.generate_meta_button = True

    st.header("")
    st.header("1. Generate Meta data")
    if 'generate_meta_button' not in st.session_state:
        st.session_state.generate_meta_button = False

    st.button("Generate Meta data", on_click=on_click_generate_meta_button)
    if st.session_state.generate_meta_button:
        with st.spinner("Now loading.."):
            with st_stdout("code"):
                run_command(join(PATH.script, "download_dataset.sh"), cwd=PATH.root)
            st.success("[SUCCESS] Download Dataset")

            for data_id in ('family', 'sample_child', 'sample_adult'):
                with st_stdout("code"):
                    print(f"Data ID: {data_id}")
                    generate_meta(data_id)
            st.success("[SUCCESS] Generate Meta data")
