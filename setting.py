"""
###################################################################
###################################################################

            This py provides instructions for 
        generating data, running or testing models 

###################################################################
###################################################################
"""

"""
---------------------Log setting-------------------------
# uncomment some codes related to config.log in main.py 

# open tensorboard 
tensorboard --logdir=./tensorboard --host=127.0.0.1
--------------------------------------------------------- 
"""

"""
---------------------Generating test data----------------
By changing setting of has_noise and SNR can generate different type of testing data and sensing matrix A.       
Commands:

python main.py --M 250 --N 500 --support 0.1 --task_type gd
python main.py --M 250 --N 500 --support 0.1 --has_noise True --SNR 20 --task_type gd 
python main.py --M 250 --N 500 --support 0.1 --has_noise True --SNR 10 --task_type gd
python main.py --M 250 --N 500 --support 0.1 --has_noise True --SNR 30 --task_type gd


--M, Dimension of matrix A, which is the sensing matrix
--N, Columns of matrix A, which is the sensing matrix
--support, The proportion supports of sparse solvers
--task_type, "gd" means generating data
--has_noise, Whether has_noise is added
--SNR, Strength of noises in measurements

--------------------------------------------------------- 
"""


"""
---------------------Testing----------------
The main differences between training and testing is that test setting has to add 
--task_type ste
--test True
the rests are the same.

## basic settings:
--M 250
--N 500
--support 0.1
-l 0.4 
-T 16

# testing settings:
--task_type ste
--test True
--tbs 64 
--vbs 1000
--net LISTA_en_de 
--scope LISTA_en_de 
--exp_id 0

# has_noise 
--has_noise True 
--SNR 20

# Nesterov's acceleration
--is_na True

# support selection 
--has_ss True 
-p 1.2 
-maxp 13

# weight shared/is_unshared, --numbers 16, indicates that weights are shared between every 16 layers, 
--is_unshared True
--numbers 16                                                        

---------------------------------------------ISTA--------------------------------------------------------------------------------------
####----ISTA
python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --numbers 16 \
--task_type ste --test True --net ISTA_en_de --scope ISTA_en_de --exp_id 0 

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --numbers 16 \
--has_noise True --SNR 20 \
--task_type ste --test True --net ISTA_en_de --scope ISTA_en_de --exp_id 0

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --numbers 16 \
--has_noise True --SNR 10 \
--task_type ste --test True --net ISTA_en_de --scope ISTA_en_de --exp_id 0

---------------------------------------------FISTA--------------------------------------------------------------------------------------
####----FISTA
python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --numbers 16 \
--task_type ste --test True --net FISTA_en_de --scope FISTA_en_de --exp_id 0 

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --numbers 16 \
--has_noise True --SNR 20 \
--task_type ste --test True --net FISTA_en_de --scope FISTA_en_de --exp_id 0 

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --numbers 16 \
--has_noise True --SNR 10 \
--task_type ste --test True --net FISTA_en_de --scope FISTA_en_de --exp_id 0


---------------------Training----------------
To train models, you have to modify following settings:
# basic settings:
--M 250
--N 500
--support 0.1
-l 0.4 
-T 16

# training setting
--task_type str
--tbs 64 
--vbs 1000
--net LISTA_en_de 
--scope LISTA_en_de 
--exp_id 0

# has_noise 
--has_noise True 
--SNR 20

# Nesterov's acceleration
--is_na True

# support selection 
--has_ss True 
-p 1.2 
-maxp 13

# weight shared/is_unshared, --numbers 16, indicates that weights are shared between every 16 layers, 
--is_unshared True
--numbers 16

---------------------------------------------LISTA-shared-------------------------------------------------------------------------------------
####----LISTA
python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --numbers 16 \
--task_type str --net LISTA_en_de --scope LISTA_en_de --exp_id 0 

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --numbers 16 \
--has_noise True --SNR 20 \
--task_type str --net LISTA_en_de --scope LISTA_en_de --exp_id 0

---------------------------------------------LISTA-not-shared-------------------------------------------------------------------------------------
####----LISTA
python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 1 \
--task_type str --net LISTA_en_de --scope LISTA_en_de --exp_id 0 

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 1 \
--has_noise True --SNR 20 \
--task_type str --net LISTA_en_de --scope LISTA_en_de --exp_id 0

---------------------------------------------LISTA-tied-------------------------------------------------------------------------------------
####----LISTA-16
python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--task_type str --net LISTA_en_de --scope LISTA_en_de --exp_id 0 --local3----

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --has_noise True --SNR 20 \
--is_unshared True --numbers 16 \
--task_type str --net LISTA_en_de --scope LISTA_en_de --exp_id 0 --local2--

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --has_noise True --SNR 10 \
--is_unshared True --numbers 16 \
--task_type str --net LISTA_en_de --scope LISTA_en_de --exp_id 0

---------------------------------------------LISTA-is_na-------------------------------------------------------------------------------------

####----LISTA-is_na-16
python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--is_na True \
--task_type str --net LISTA_en_de --scope LISTA_en_de --exp_id 0

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--is_na True --has_noise True --SNR 20 \
--task_type str --net LISTA_en_de --scope LISTA_en_de --exp_id 0

---------------------------------------------TsLISTA-------------------------------------------------------------
####----TsLISTA

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 1 \
--task_type str --net TsLISTA_en_de --scope TsLISTA_en_de --exp_id 0

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 1 \
--has_noise True --SNR 20 \
--task_type str --net TsLISTA_en_de --scope TsLISTA_en_de --exp_id 0

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 1 \
--has_noise True --SNR 10 \
--task_type str --net TsLISTA_en_de --scope TsLISTA_en_de --exp_id 0


---------------------------------------------LISTA-CP-------------------------------------------------------------------------------------
####----LISTA-CP
python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 1 \
--task_type str --net LISTA_CP_en_de --scope LISTA_CP_en_de --exp_id 0 --local4----

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --has_noise True --SNR 20 \
--is_unshared True --numbers 1 \
--task_type str --net LISTA_CP_en_de --scope LISTA_CP_en_de --exp_id 0 --local

---------------------------------------------LISTA-CP-is_na-------------------------------------------------------------------------------------

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 1 \
--is_na True \
--task_type str --net LISTA_CP_en_de --scope LISTA_CP_en_de --exp_id 0


python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 1 \
--is_na True --has_noise True --SNR 20 \
--task_type str --net LISTA_CP_en_de --scope LISTA_CP_en_de --exp_id 0


---------------------------------------------LISTA-CP-ss-------------------------------------------------------------------------------------
####----LISTA-CP
python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 1 \
--has_ss True -p 1.2 -maxp 13 \
--task_type str --net LISTA_CP_en_de --scope LISTA_CP_en_de --exp_id 0 --local2----

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --has_noise True --SNR 20 \
--is_unshared True --numbers 1 --has_ss True -p 1.2 -maxp 13 \
--task_type str --net LISTA_CP_en_de --scope LISTA_CP_en_de --exp_id 0 --local2--


python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --has_noise True --SNR 10 \
--is_unshared True --numbers 1 --has_ss True -p 1.2 -maxp 13 \
--task_type str --net LISTA_CP_en_de --scope LISTA_CP_en_de --exp_id 0 --local2--


---------------------------------------------LFISTA-------------------------------------------------------------------------------------
####----LFISTA
python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 1 \
--task_type str --net LFISTA_en_de --scope LFISTA_en_de --exp_id 0 --local2----

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --has_noise True --SNR 20 \
--is_unshared True --numbers 1 \
--task_type str --net LFISTA_en_de --scope LFISTA_en_de --exp_id 0

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --has_noise True --SNR 10 \
--is_unshared True --numbers 1 \
--task_type str --net LFISTA_en_de --scope LFISTA_en_de --exp_id 0


---------------------------------------------LsLISTA--------------------------------------------------------------------------------------
####----LsLISTA
python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 1 \
--task_type str --net LsLISTA_en_de --scope LsLISTA_en_de --exp_id 0 --local6----

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 1 \
--has_noise True --SNR 20 \
--task_type str --net LsLISTA_en_de --scope LsLISTA_en_de --exp_id 0

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 1 \
--has_noise True --SNR 10 \
--task_type str --net LsLISTA_en_de --scope LsLISTA_en_de --exp_id 0


---------------------------------------------LsLISTA-tied-------------------------------------------------------------------------------------
python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--task_type str --net LsLISTA_en_de --scope LsLISTA_en_de --exp_id 0 --local3----

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--has_noise True --SNR 20 \
--task_type str --net LsLISTA_en_de --scope LsLISTA_en_de --exp_id 0 --local6--

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--has_noise True --SNR 10 \
--task_type str --net LsLISTA_en_de --scope LsLISTA_en_de --exp_id 0


---------------------------------------------LsLISTA-tied-is_na-------------------------------------------------------------------------------------

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--is_na True \
--task_type str --net LsLISTA_en_de --scope LsLISTA_en_de --exp_id 0

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--is_na True --has_noise True --SNR 20 \
--task_type str --net LsLISTA_en_de --scope LsLISTA_en_de --exp_id 0


---------------------------------------------LsLISTA-t-ss--------------------------------------------------------------------------------------
####----LsLISTA
python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--has_ss True -p 1.2 -maxp 13 \
--task_type str --net LsLISTA_en_de --scope LsLISTA_en_de --exp_id 0 --local6----

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--has_ss True -p 1.2 -maxp 13 --has_noise True --SNR 20 \
--task_type str --net LsLISTA_en_de --scope LsLISTA_en_de --exp_id 0 --local6----


python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--has_ss True -p 1.2 -maxp 13 --has_noise True --SNR 10 \
--task_type str --net LsLISTA_en_de --scope LsLISTA_en_de --exp_id 0


---------------------------------------------LM_FISTA---------------------------------------------------------------------------------------
####----LM_FISTA
python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--task_type str --net LM_FISTA_en_de --scope LM_FISTA_en_de --exp_id 0 --local----

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--has_noise True --SNR 20 \
--task_type str --net LM_FISTA_en_de --scope LM_FISTA_en_de --exp_id 0

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
-p 1.2 -maxp 13 --has_noise True --SNR 10 \
--task_type str --net LM_FISTA_en_de --scope LM_FISTA_en_de --exp_id 0  ---local5---

---------------------------------------------LM_FISTA-ss--------------------------------------------------------------------------------------

####----LM_FISTA-ss
python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--has_ss True -p 1.2 -maxp 13 \
--task_type str --net LM_FISTA_en_de --scope LM_FISTA_en_de --exp_id 0 --local4----

####----LM_FISTA-ss
python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--has_ss True -p 1.2 -maxp 13 --has_noise True --SNR 20 \
--task_type str --net LM_FISTA_en_de --scope LM_FISTA_en_de --exp_id 0 --local5----

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--has_ss True -p 1.2 -maxp 13 --has_noise True --SNR 10 \
--task_type str --net LM_FISTA_en_de --scope LM_FISTA_en_de --exp_id 0  ---local5---


---------------------------------------------LM_FISTA_cp_---------------------------------------------------------------------------------------
####----LM_FISTA_cp_
python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 1 \
--task_type str --net LM_FISTA_cp_en_de --scope LM_FISTA_cp_en_de --exp_id 0 --local----

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 1 \
--has_noise True --SNR 20 \
--task_type str --net LM_FISTA_cp_en_de --scope LM_FISTA_cp_en_de --exp_id 0

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 1 \
-p 1.2 -maxp 13 --has_noise True --SNR 10 \
--task_type str --net LM_FISTA_cp_en_de --scope LM_FISTA_cp_en_de --exp_id 0  ---local5---


####----LM_FISTA_cp_-ss
python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 1 \
--has_ss True -p 1.2 -maxp 13 \
--task_type str --net LM_FISTA_cp_en_de --scope LM_FISTA_cp_en_de --exp_id 0 --local4----


---------------------------------------------TiLISTA---------------------------------------------------------------------------------------
####----TiLISTA
python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--task_type str --net TiLISTA_en_de --scope TiLISTA_en_de -p 1.2 -maxp 13 --exp_id 0


python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--has_noise True --SNR 20 \
--task_type str --net TiLISTA_en_de --scope TiLISTA_en_de -p 1.2 -maxp 13 --exp_id 0 --local5----

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--has_noise True --SNR 10 \
--task_type str --net TiLISTA_en_de --scope TiLISTA_en_de -p 1.2 -maxp 13 --exp_id 0


---------------------------------------------LAMP--------------------------------------------------------------------------------------
####----LAMP
python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--task_type str --net LAMP_en_de --scope LAMP_en_de --exp_id 0 --local2----

####----LAMP
python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--has_noise True --SNR 20 \
--task_type str --net LAMP_en_de --scope LAMP_en_de --exp_id 0 --local4----

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--has_noise True --SNR 10 \
--task_type str --net LAMP_en_de --scope LAMP_en_de --exp_id 0


---------------------------------------------ALISTA---------------------------------------------------------------------------------------
 ####----ALISTA
python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--task_type str --net ALISTA_en_de --scope ALISTA_en_de -p 1.2 -maxp 13 --exp_id 0 --local6----

####----ALISTA
python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--has_noise True --SNR 20 \
--task_type str --net ALISTA_en_de --scope ALISTA_en_de -p 1.2 -maxp 13 --exp_id 0 --local6----

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--has_noise True --SNR 10 \
--task_type str --net ALISTA_en_de --scope ALISTA_en_de -p 1.2 -maxp 13 --exp_id 0

---------------------------------------------ALISTA-na--------------------------------------------------------------------------------------
#
python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--is_na True \
--task_type str --net ALISTA_en_de --scope ALISTA_en_de -p 1.2 -maxp 13 --exp_id 0

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--is_na True --has_noise True --SNR 20 \
--task_type str --net ALISTA_en_de --scope ALISTA_en_de -p 1.2 -maxp 13 --exp_id 0

---------------------------------------------ALISTA-ss--------------------------------------------------------------------------------------
####----ALISTA
python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--ss True \
--task_type str --net ALISTA_en_de --scope ALISTA_en_de -p 1.2 -maxp 13 --exp_id 0

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--has_ss True --has_noise True --SNR 20 \
--task_type str --net ALISTA_en_de --scope ALISTA_en_de -p 1.2 -maxp 13 --exp_id 0

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--has_ss True --has_noise True --SNR 10 \
--task_type str --net ALISTA_en_de --scope ALISTA_en_de -p 1.2 -maxp 13 --exp_id 0


---------------------------------------------LM_ALISTA---------------------------------------------------------------------------------------
####----LM_ALISTA
python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
-p 1.2 -maxp 13 \
--task_type str --net LM_ALISTA_en_de --scope LM_ALISTA_en_de --exp_id 0 --local----

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--has_noise True --SNR 20 -p 1.2 -maxp 13 \
--task_type str --net LM_ALISTA_en_de --scope LM_ALISTA_en_de --exp_id 0

python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
-p 1.2 -maxp 13 --has_noise True --SNR 10 \
--task_type str --net LM_ALISTA_en_de --scope LM_ALISTA_en_de --exp_id 0  ---local5---


---LM-ALISTA-ss
python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
-p 1.2 -maxp 13 --has_ss True \
--task_type str --net LM_ALISTA_en_de --scope LM_ALISTA_en_de --exp_id 0

---------------------------------------------GLISTA-exp-ss-------------------------------------------------------------------------------------
####----GLISTA

python main.py --M 250 --N 500 --support 0.1 --gain_fun exp --has_ss True -l 0.4 -T 16 --tbs 64 --vbs 1000 \
--is_unshared True --numbers 1 \
--task_type str --net GLISTA_en_de -p 1.2 -maxp 13 --scope GLISTA_en_de --exp_id 0 --local2--

python main.py --M 250 --N 500 --support 0.1 --gain_fun exp --has_ss True -l 0.4 -T 16 --tbs 64 --vbs 1000 \
--is_unshared True --numbers 1 --has_ss True --has_noise True --SNR 20 \
--task_type str --net GLISTA_en_de -p 1.2 -maxp 13 --scope GLISTA_en_de --exp_id 0 --local2--

python main.py --M 250 --N 500 --support 0.1 --gain_fun exp --has_ss True -l 0.4 -T 16 --tbs 64 --vbs 1000 \
--is_unshared True --numbers 1 --has_ss True --has_noise True --SNR 10 \
--task_type str --net GLISTA_en_de -p 1.2 -maxp 13 --scope GLISTA_en_de --exp_id 0

"""


