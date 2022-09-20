import pandas as pd
import re

df = pd.read_excel(r'C:\Users\vimkumar\Downloads\Project\User_data.xlsx')

Option_1='Login'
Option_2='Register'
Option_3='Forgot_Password'
User_Input=input('Login/Register/Forgot_Password: ')

if User_Input==Option_2:
    Username=input("Enter Username: ")
    a = Username.find('@')
    b = Username.find('.')
    if b>a and a+1!=b and Username[0].isnumeric()==False and Username[0].isalnum()==True:
        print('User accepted')
        if len(df[(df['User']==Username )])==1:
            print('User already taken')
        else:    
            Password=input("Enter Password: ")   
            pattern1="^.*(?=.{5,16})(?=.*\d)(?=.*[a-z])(?=.*[A-Z])(?=.*[@#$%^&+=]).*$"
            if len(Password)>5 and len(Password)<16 and re.findall(pattern1,Password): 
                tmp = pd.DataFrame({'User':[Username],'Pass':[Password]})
                df = df.append(tmp)    
                df.to_excel(r'C:\Users\vimkumar\Downloads\Project\User_data.xlsx',index=False)
            else:
                print("Please Enter Valid Password")
    else:
        print('not a valid username')      
        
elif User_Input==Option_1:
    Username=input("Enter Username: ")
    Password=input("Enter Password: ")
   
    if len(df[(df['User']==Username )&(df['Pass']==Password)])==1:
        print('Login successful')
    else:
        print('Username or password is wrong')
                      
elif User_Input==Option_3:
    O4='Retrieve'
    O5='Create_New'
    User_Option_2=input('Retrieve/Create_New')
    if User_Option_2==O4:
        Username=input("Enter Username: ")
        print('Password :',df[df['User']==Username]['Pass'].iloc[0])
    elif User_Option_2==O5:
        Username=input("Enter Username: ")
        if len(df[(df['User']==Username )])==1:        
            Password=input("Enter Password: ")
            df.loc[(df['User']==Username ),'Pass'] = Password
            df.to_excel(r'C:\Users\vimkumar\Downloads\Project\User_data.xlsx',index=False)
        else:
            print('user not found')

        