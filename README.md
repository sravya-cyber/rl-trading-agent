
<p align="center">
<img height="200px" src="https://user-images.githubusercontent.com/78701055/155994640-18c37b08-f6dc-493b-8116-c8b46f94a46b.png" alt="Loading img..."/>
<!-- <img height="200px"  src="https://user-images.githubusercontent.com/78701055/155997992-04602a01-5dd9-41c3-a563-597fabf7d8d1.jpg" alt="Loading img..."/> -->
</p>
<h1 align="center"> Spardha'24 </h1>

Spardha is the annual sports festival of IIT (BHU), Varanasi. In its glorious history of over 36 years, Spardha has grown to become the largest and one of the most awaited sports festivals of northern India, where athletic competition is drawn from throughout the country.

## Before You Begin
Before you begin, we recommend you read about the basic building blocks that assemble this website:
### Before making PR things to be check
- Get pull from the master to resolve conflicts if there
- Get pull from dev to resolve conflicts if there
- Don't make PR direct to the production branch(master), instead make PR to development branch(dev)
- After making PR wait and check, do all checks passed, and PR built successfully on Deployment Server
### For Project setup
- You can download this repo and run it locally but Don't put it on your GitHub.
- Download the repo and make a new branch (the name should be related to the feature you are working on). Then after making code changes push it on the Spardha repo and make pr.
- Don't make any changes on master (locally also). Whatever you want to do, make a new branch for that.
- all branches after your PR Accept/Reject are automatically deleted.
- all PR would be made on the dev branch, not to the direct master.

## Prerequisites
Make sure you have installed all of the following prerequisites on your development machine:
- Git - Download & Install Git. OSX and Linux machines typically have this already installed.
- VS Code - Download & Install VS Code, one of the most popular code - editor developed by Microsoft.

### Frontend :
- ReactJs,
   
### Backend:
- Django, GoogleAPI's,MySQL


## Get Started
Once you've downloaded the boilerplate and installed all the prerequisites, you're just a few steps away from starting to run the website

### Get started with Frontend
- Open the Frontend folder (Spardha-23 Project/frontend/) and open the folder in VS Code.
- Open a new terminal by clicking the command (Ctrl + Shift + `) or clicking on Terminal : New Terminal in the nav-menu
- To install the dependencies, run this in the application folder from the command line:
- ```
  npm install
  ```
- Create a .env file in the current directory of the project using the touch command in your terminal
- ```bash
  touch .env
  ```
- Get the .env variables' names from the .env.example file and paste them into the .env file
- Remember to paste all the values into the .env file
- To run the react - app, run the following command
- ```bash
  npm start
  ```
- The server will listen on PORT 3000 ( localhost:3000 )

### Get started with Backend


### Working with repo:
1. Clone the repo in which you must work in your system.
2. Create or do your work in another branch with a relevant name.
3. After completing your work, commit your changes and push them.
4. Create a pull request (PR).

### Setting up the project(Backend) :

1. Creating a virtual environment:
```bash
python -m venv venv
```
2. Activate a virtual environment

Linux:
```bash
source venv/bin/activate
```
Windows:
```cmd
./venv/Scripts/activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Move to the main folder: 
```bash
cd Spardha
```
5. Apply migrations: 
```bash
python manage.py migrate
```
6. Collect static files : 
```bash
python manage.py collectstatic
```
7. Start the development server: 
```bash
python manage.py runserver
```

