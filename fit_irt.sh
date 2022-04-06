echo "----------------------------------------------------------------IRT Fit Start----------------------------------------------------------------"
# Country
py-irt train 1pl --lr 0.05 --epochs 5000 --priors hierarchical --device cuda /home/haohang/research/CL_upload/total_data/model_judgement/Country.jsonlines /home/haohang/research/CL_upload/total_data/fitted_IRT/Country > /home/haohang/research/CL_upload/total_data/fitted_IRT/Country/Country.txt
# Perspective
py-irt train 1pl --lr 0.05 --epochs 5000 --priors hierarchical --device cuda /home/haohang/research/CL_upload/total_data/model_judgement/Perspective.jsonlines /home/haohang/research/CL_upload/total_data/fitted_IRT/Perspective > /home/haohang/research/CL_upload/total_data/fitted_IRT/Perspective/Perspective.txt
# Population
py-irt train 1pl --lr 0.05 --epochs 5000 --priors hierarchical --device cuda /home/haohang/research/CL_upload/total_data/model_judgement/Population.jsonlines /home/haohang/research/CL_upload/total_data/fitted_IRT/Population > /home/haohang/research/CL_upload/total_data/fitted_IRT/Population/Population.txt
# Intervention
py-irt train 1pl --lr 0.05 --epochs 5000 --priors hierarchical --device cuda /home/haohang/research/CL_upload/total_data/model_judgement/Intervention.jsonlines /home/haohang/research/CL_upload/total_data/fitted_IRT/Intervention > /home/haohang/research/CL_upload/total_data/fitted_IRT/Intervention/Intervention.txt
# Study Period
py-irt train 1pl --lr 0.05 --epochs 5000 --priors hierarchical --device cuda /home/haohang/research/CL_upload/total_data/model_judgement/'Study Period.jsonlines' /home/haohang/research/CL_upload/total_data/fitted_IRT/'Study Period' > /home/haohang/research/CL_upload/total_data/fitted_IRT/'Study Period'/'Study Period.txt'
# Sample Size
py-irt train 1pl --lr 0.05 --epochs 5000 --priors hierarchical --device cuda /home/haohang/research/CL_upload/total_data/model_judgement/'Sample Size.jsonlines' /home/haohang/research/CL_upload/total_data/fitted_IRT/'Sample Size' > /home/haohang/research/CL_upload/total_data/fitted_IRT/'Sample Size'/'Sample Size.txt'
echo "----------------------------------------------------------------IRT Fit End----------------------------------------------------------------"
