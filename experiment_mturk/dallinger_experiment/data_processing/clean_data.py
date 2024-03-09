import pandas as pd
import json 

# The directory of the experiment to clean the data for and the output filename to use
experiment_string = '' # put here
output_filename = '' # put here

# Load in info table and convert to pandas df
raw_info_table = pd.read_csv("../data/"+experiment_string+"/data/info.csv")

info_dict = []
for index, row in raw_info_table.iterrows():
    if row['contents'][0] == '{':
        info_dict.append(json.loads(row['contents']))
info_table = pd.DataFrame(info_dict)

# Load in participant table
participant_table =  pd.read_csv("../data/"+experiment_string+"/data/participant.csv")

# Merge to get participant status, drop id column
all_data = pd.merge(info_table,participant_table[['id','status']],left_on='participant_id',right_on='id').drop('id',axis=1)

# Exclude returned, abanonded, and did not attend participants
all_data = all_data[all_data.status.isin(['approved','did_not_attend'])]

# Make a new column called "response accepted" for ease of use and drop status column
all_data = all_data[all_data['status'] == 'approved']
all_data = all_data.drop('status',axis=1)

# Save
all_data.to_csv('../cleaned_data/final_experiment/' + output_filename + '.csv')