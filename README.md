# Parking-Spaces-Classifier
It's a project, concluding the Neural Networks Course taken in University of Wroclaw in 2026. Creators of following project are: Kornel Orawczak, Marcin Mularczyk, Mateusz Matyskiel, Jan Lachowski. 

## UV environment 
In this project we will work on a shared uv environment. All of us worked on macos system, so all the commands are written with that in mind 
### Initial setup 
To first use uv, you need to download it using
```bash
brew install uv
```
### Activating the env 
In order to activate the project environment you simply write
```bash
source .venv/bin/activate
```
### Adding a package (for collaborators)
If you want to add a new package to the shared uv, write:
```bash
uv add [name_of_package]
```
## Working with dataset (for collaborators)
In order to do the work with dataset you need to first log in to your kaggle account and create an API key for yourself (Settings -> API -> Generate token). It will start with "KGAT...", save it somewhere safe and copy to cliboard. 

Then you need to save it to zsh, using 
```bash
echo 'export KAGGLE_API_TOKEN="[YOUR_TOKEN]"' >> ~/.zshrc
source ~/.zshrc
```
Then you can just use the created makefile to have the dataset yourself
```bash
make fetch-data
```

