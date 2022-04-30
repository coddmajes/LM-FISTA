'''
---------------------------------------------------------------------------------------------------------------------------------------
For face recognition


---------------------------------------------ISTA--------------------------------------------------------------------------------------
####----ISTA-16
python main_cf.py -l 0.4 -T 100 --net ISTA_en_de --scope ISTA_en_de --en_de True \
--data_type Yale --feature_type random --classify_type SRC --feature_shape 120 --task_type fte --test True --exp_id 0

---------------------------------------------FISTA--------------------------------------------------------------------------------------
####----FISTA-16
python main_cf.py -l 0.4 -T 100 --net FISTA_en_de --scope FISTA_en_de --en_de True \
--data_type Yale --feature_type random --classify_type SRC --feature_shape 120 --task_type fte --test True --exp_id 0


---------------------------------------------LISTA-nums-------------------------------------------------------------------------------------
####----LISTA-16
python main_cf.py -l 0.4 -T 16 --tbs 64 --vbs 604 --tr_samples 1205 --val_samples 604 --unshared True --numbers 16 \
--net LISTA_en_de --scope LISTA_en_de --en_de True --better_wait 100 --init_lr 5e-4 \
--data_type Yale --feature_type random --classify_type SRC --feature_shape 120 --task_type ftr --exp_id 0


---------------------------------------------TsLISTA_en_de----------------------------------------------------------------------------

python main_cf.py -l 0.4 -T 16 --tbs 64 --vbs 604 --tr_samples 1205 --val_samples 604 --unshared True --numbers 1 \
--net TsLISTA_en_de --scope TsLISTA_en_de --en_de True --better_wait 100 --init_lr 5e-4 \
--data_type Yale --feature_type random --classify_type SRC --feature_shape 120 --task_type ftr --exp_id 0


---------------------------------------------LsLISTA------SRC--------------------------------------------------------------------------------
####----LsLISTA----
python main_cf.py -l 0.4 -T 16 --tbs 64 --vbs 604 --tr_samples 1205 --val_samples 604 --unshared True --numbers 1 \
--net LsLISTA_en_de --scope LsLISTA_en_de --en_de True --better_wait 100 --init_lr 5e-4 \
--data_type Yale --feature_type random --classify_type SRC --feature_shape 120 --task_type ftr --exp_id 0

####----LsLISTA-t---
python main_cf.py -l 0.4 -T 16 --tbs 64 --vbs 604 --tr_samples 1205 --val_samples 604 --unshared True --numbers 16 \
--net LsLISTA_en_de --scope LsLISTA_en_de --en_de True --better_wait 100 --init_lr 5e-4 \
--data_type Yale --feature_type random --classify_type SRC --feature_shape 120 --task_type ftr --exp_id 0


---------------------------------------------TiLISTA--------------------------------------------------------------------------------------
####----TiLISTA
python main_cf.py -l 0.4 -T 16 --tbs 64 --vbs 604 --tr_samples 1205 --val_samples 604 --unshared True --numbers 16 \
--net TiLISTA_en_de --scope TiLISTA_en_de --en_de True --better_wait 100 --init_lr 5e-4 -p 1.2 -maxp 13 \
--ss True \
--data_type Yale --feature_type random --classify_type SRC --feature_shape 120 --task_type ftr --exp_id 0


---------------------------------------------LM_FISTA--------------------------------------------------------------------------------------
####----LM_FISTA
python main_cf.py -l 0.4 -T 16 --tbs 64 --vbs 604 --tr_samples 1205 --val_samples 604 --unshared True --numbers 16 \
--net LM_FISTA_en_de --scope LM_FISTA_en_de --en_de True --better_wait 100 --init_lr 5e-4 -p 1.2 -maxp 13 \
--data_type Yale --feature_type random --classify_type SRC --feature_shape 120 --task_type ftr --exp_id 0


---------------------------------------------------------------------------------------------------------------------------------------

'''