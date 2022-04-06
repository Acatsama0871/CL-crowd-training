echo "----------------------------------------------------------------IRT Fit Start----------------------------------------------------------------"
# Country
py-irt train 1pl --lr 0.1 --epochs 5000 --priors hierarchical --device cuda total_data/model_judgement/Country.jsonlines total_data/fitted_IRT/Country > total_data/fitted_IRT/Country/Country.txt
# Perspective
py-irt train 1pl --lr 0.1 --epochs 5000 --priors hierarchical --device cuda total_data/model_judgement/Perspective.jsonlines total_data/fitted_IRT/Perspective > total_data/fitted_IRT/Perspective/Perspective.txt
# Population
py-irt train 1pl --lr 0.1 --epochs 5000 --priors hierarchical --device cuda total_data/model_judgement/Population.jsonlines total_data/fitted_IRT/Population > total_data/fitted_IRT/Population/Population.txt
# Intervention
py-irt train 1pl --lr 0.1 --epochs 5000 --priors hierarchical --device cuda total_data/model_judgement/Intervention.jsonlines total_data/fitted_IRT/Intervention > total_data/fitted_IRT/Intervention/Intervention.txt
# Study Period
py-irt train 1pl --lr 0.1 --epochs 5000 --priors hierarchical --device cuda total_data/model_judgement/'Study Period.jsonlines' total_data/fitted_IRT/'Study Period' > total_data/fitted_IRT/'Study Period'/'Study Period.txt'
# Sample Size
py-irt train 1pl --lr 0.1 --epochs 5000 --priors hierarchical --device cuda total_data/model_judgement/'Sample Size.jsonlines' total_data/fitted_IRT/'Sample Size' > total_data/fitted_IRT/'Sample Size'/'Sample Size.txt'
echo "----------------------------------------------------------------IRT Fit End----------------------------------------------------------------"
