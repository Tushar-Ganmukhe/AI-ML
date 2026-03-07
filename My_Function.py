import pandas as pd

def show_encoded_data(encoded_array, encoder):
    df1 = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out()
    )
    print(df1)