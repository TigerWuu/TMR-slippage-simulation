# TMR-slippage-simulation
This is a simulation of TMR robot system with slippage compensation


## Contents
- [Installation](#Installatio)
- [Usage](#Usage)
  - [slippage train](#slippage-train)
  - [slippage test](#slippage-test)

## Installation
- Clone the package .

      git clone https://github.com/TigerWuu/TMR-slippage-simulation.git
      
- package

  download with pip(recommand) or conda
  
      pip install tensorfow
      pip install matplotlib 
## Usage
- first, run "slippage_train.py" to get the NN model "slippage_predict"
 
- Next, run "slippage_test.py" to load the NN model "slippage_predict" and verify the compensation result
      
### slippage train.py


### slippage test.py
- There are four trajectory simulation ,saparately

 1. reference trajectory
 2. trajectory without slippage compensation
 3. trajectory with slippage compensation offline
 4. trajectory with slippage compensation online

- "ra" can be changed to modified the slippage when the time reach the time_step/3

      # ra = 1 ---> slippage remains
      # ra > 1 ---> slippage = slippage * ra
- "online" can switch the training status

      # online = True ---> online & offline
      # online = False ---> offline only
