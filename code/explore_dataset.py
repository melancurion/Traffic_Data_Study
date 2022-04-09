import pandas as pd
import numpy as np
# import seaborn as sns
# import sklearn
import os
# import psycopg2

# For naive Bayes
# from sklearn.naive_bayes import GaussianNB
# SVM
# from sklearn import svm
# import pickle
# import xlswriter




###################################################
##              Function definitions              #
###################################################

# Ascertain which keys exist in a given s3 bucket with a given prefix
# def get_keys(Bucket, Prefix, S3Connection, S3Client, n=2000, show_progress=False):
    


def get_dataset_stats(df, exclude=["commission_date","physical_safety_test","damaged","id"], round_digits=3):
    # Get list of columns with 'object' datatype
    # Exclude ETL columns
    # Handle "exclude" columns differently to avoid internal errors.
    exclude_columns = ['etl_proc_wid','w_insert_dt','w_update_dt',
                       'effective_from_dt','effective_to_dt','current_flg',
                       'delete_flg'] + exclude
    include_columns = list(df.columns)
    for element in exclude_columns:
        if element in include_columns:
            include_columns.remove(element)
    
    
    types_df = pd.DataFrame(df[include_columns].dtypes).rename({0:"dtype"}, axis=1)
    obj_list = df[include_columns].select_dtypes(include=['object']).columns.to_list()
    # local function to determine new datatypes for object columns:
    def get_obj_type(df, var):
        type = ""
        if df[var].isnull().all():
            type = 'none'
        else:
            try:
                df[var].astype("datetime64[ns]")
                type = 'datetime64[ns]'
            except:
                try:
                    df[var].astype("int64")
                    type = 'int64'
                except:
                    try:
                        df[var].astype("float64")
                        type = 'float64'  
                    except:
                        try:
                            df[var].astype("str")
                            type = 'str'
                        except:
                            type = 'str'
                            pass
        return type    
    
    for col in obj_list:
        new_type = get_obj_type(df, col)
        print("{} new type is {}".format(col, new_type))
        types_df["dtype"][col] = new_type
    
    
    dg = pd.DataFrame(np.nan, index=include_columns, 
                      columns=["type","excluded","non_null_count","nullcnt","%nulls","min",
                               "pcntl25","median","mean","pcntl75","max",
                               "stddev","str-maxlen","str-meanlen","str-medianlen","str-minlen","ndistinct"])
    
    try:
        for varname in include_columns:
            if str(types_df["dtype"][varname]) in ["datetime64[ns, psycopg2.tz.FixedOffsetTimezone(offset=0, name=None)]"]:
                df[varname] = df[varname].dt.tz_localize(None)
                types_df["dtype"][varname] = "datetime64[ns]"
                
            dg.type.loc[varname] = types_df.loc[varname, "dtype"]
            if dg.type.loc[varname] == 'float64':
                df[varname] = df[varname].astype('float')
            elif dg.type.loc[varname] == 'int64':
                df[varname] = df[varname].astype('int')
            if varname in exclude:
                dg.loc[varname,"excluded"] = 'Yes'
            else:
                dg.loc[varname,"excluded"] = 'No'
            # print("Var: {}  Type: {}".format(varname, dg.type.loc[varname]))
            dg.loc[varname,"ndistinct"] = df[varname].nunique()

            if types_df["dtype"][varname] in ["float64","int64","int32","datetime64[ns]","str","none"]:
                dg.loc[varname, "nullcnt"] = df[varname].isna().sum().astype(int)
                totalcnt = df[varname].count().astype(int) + dg.loc[varname, "nullcnt"]
                dg.loc[varname,"non_null_count"] = totalcnt - dg.loc[varname, "nullcnt"]
                dg.loc[varname,"%nulls"] = (100.0 * dg.loc[varname,"nullcnt"] / totalcnt).round(decimals=2)
            if types_df["dtype"][varname] in ["datetime64[ns]"]:
                dg.loc[varname,"min"] = df[varname].dropna().min()
                dg.loc[varname,"max"] = df[varname].dropna().max()
                dg.loc[varname,"mean"] = df[varname].dropna().mean()
            elif types_df["dtype"][varname] in ["float64","int64","int32"]:
                dg.loc[varname,"min"] = round(df[varname].dropna().min(), round_digits)
                dg.loc[varname,"max"] = round(df[varname].dropna().max(), round_digits)
                dg.loc[varname,"mean"] = round(df[varname].dropna().mean(), round_digits)                
            if types_df["dtype"][varname] in ["float64","int64","int32"]:
                dg.loc[varname,"pcntl25"] = round(df[varname].dropna().quantile(0.25), round_digits)
                dg.loc[varname,"median"] = round(df[varname].dropna().quantile(0.5), round_digits) 
                dg.loc[varname,"pcntl75"] = round(df[varname].dropna().quantile(0.75), round_digits)
                dg.loc[varname,"stddev"] = round(df[varname].dropna().std(), round_digits)
            if types_df["dtype"][varname] in ["str"] and varname not in exclude:
                dg.loc[varname,"str-maxlen"] = df[varname].str.len().max()
                dg.loc[varname,"str-meanlen"] = df[varname].str.len().mean()
                dg.loc[varname,"str-medianlen"] = df[varname].str.len().median()
                dg.loc[varname,"str-minlen"] = df[varname].str.len().min()
                           
    except:
        print("Failed iteration on variable: {}".format(varname))
    
    return dg



###################################################
##              Parameters                        #
###################################################


data_dir = "D:/Projects/Git Root/crash-data-analysis/data"
os.chdir(data_dir)
os.getcwd()

## Load data
df = pd.read_csv("Crash_Analysis_System_(CAS)_data.csv")

## Create statistics frame
df_stats = get_dataset_stats(df)

## Find all values for low-cardinality str columns and see if we can spot anomalies
for col in df_stats[df_stats.type == 'str'][df_stats.ndistinct < 60].index.values.tolist():
    print(col + ': ')
    print('\t', end='') 
    print(df[col].unique())


# Possible values are 'Traffic Signals', 'Stop Sign', 'Give Way Sign', 'Pointsman', 'School Patrol', 'Nil' or ' N/A'.


with pd.ExcelWriter('CAS_stats_20220408.xlsx') as writer:  
    po.to_excel(writer, sheet_name='Pole')
    pi.to_excel(writer, sheet_name='Pole Inspection')
    vt.to_excel(writer, sheet_name='Vonaq Test')