import os

from yeval.task import register_task, YevalTask
from yeval.response.math_responses import get_boxed_answer

path = os.path.dirname(__file__)

letter_choices = ["A", "B", "C", "D"]

def input_text(x):
    choices = x["choices"]
    input_command = "Choose the correct answer from the options below:\nThink step by step and write your final answer in \\boxed{}.\n"
    return input_command + f"{x['question']}\n\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\n\n"

def output_text(x):
    label = letter_choices[x["answer"]]
    choices = x["choices"]
    answer = choices[x["answer"]]
    return f"{label}::{answer}"

def eval_with_postprocessing(x, y):
    gold_letter, gold_answer = y.split("::")

    ans_score = 0.0
    ans = get_boxed_answer(x)
    if ans.lower() == gold_letter.lower():
        ans_score = 1.0
    elif ans.lower() == gold_answer.lower():
        ans_score = 1.0
    elif ans.lower()[:1] == gold_letter.lower():
        ans_score = 1.0
    return ans_score

class MMLUTask(YevalTask):
    data_path="cais/mmlu"
    input_text=input_text
    output_text=output_text
    test_split="test"
    evaluation={"accuracy": eval_with_postprocessing}
    sample_agg_fn={"accuracy": lambda x: x}

@register_task("mmlu_abstract_algebra")
class MMLU_AbstractAlgebraTask(MMLUTask):
    data_name = "abstract_algebra"

@register_task("mmlu_anatomy")
class MMLU_AnatomyTask(MMLUTask):
    data_name = "anatomy"

@register_task("mmlu_astronomy")
class MMLU_AstronomyTask(MMLUTask):
    data_name = "astronomy"

@register_task("mmlu_auxiliary_train")
class MMLU_AuxiliaryTrainTask(MMLUTask):
    data_name = "auxiliary_train"

@register_task("mmlu_business_ethics")
class MMLU_BusinessEthicsTask(MMLUTask):
    data_name = "business_ethics"

@register_task("mmlu_clinical_knowledge")
class MMLU_ClinicalKnowledgeTask(MMLUTask):
    data_name = "clinical_knowledge"

@register_task("mmlu_college_biology")
class MMLU_CollegeBiologyTask(MMLUTask):
    data_name = "college_biology"

@register_task("mmlu_college_chemistry")
class MMLU_CollegeChemistryTask(MMLUTask):
    data_name = "college_chemistry"

@register_task("mmlu_college_computer_science")
class MMLU_CollegeComputerScienceTask(MMLUTask):
    data_name = "college_computer_science"

@register_task("mmlu_college_mathematics")
class MMLU_CollegeMathematicsTask(MMLUTask):
    data_name = "college_mathematics"

@register_task("mmlu_college_medicine")
class MMLU_CollegeMedicineTask(MMLUTask):
    data_name = "college_medicine"

@register_task("mmlu_college_physics")
class MMLU_CollegePhysicsTask(MMLUTask):
    data_name = "college_physics"

@register_task("mmlu_computer_security")
class MMLU_ComputerSecurityTask(MMLUTask):
    data_name = "computer_security"

@register_task("mmlu_conceptual_physics")
class MMLU_ConceptualPhysicsTask(MMLUTask):
    data_name = "conceptual_physics"

@register_task("mmlu_econometrics")
class MMLU_EconometricsTask(MMLUTask):
    data_name = "econometrics"

@register_task("mmlu_electrical_engineering")
class MMLU_ElectricalEngineeringTask(MMLUTask):
    data_name = "electrical_engineering"

@register_task("mmlu_elementary_mathematics")
class MMLU_ElementaryMathematicsTask(MMLUTask):
    data_name = "elementary_mathematics"

@register_task("mmlu_formal_logic")
class MMLU_FormalLogicTask(MMLUTask):
    data_name = "formal_logic"

@register_task("mmlu_global_facts")
class MMLU_GlobalFactsTask(MMLUTask):
    data_name = "global_facts"

@register_task("mmlu_high_school_biology")
class MMLU_HighSchoolBiologyTask(MMLUTask):
    data_name = "high_school_biology"

@register_task("mmlu_high_school_chemistry")
class MMLU_HighSchoolChemistryTask(MMLUTask):
    data_name = "high_school_chemistry"

@register_task("mmlu_high_school_computer_science")
class MMLU_HighSchoolComputerScienceTask(MMLUTask):
    data_name = "high_school_computer_science"

@register_task("mmlu_high_school_european_history")
class MMLU_HighSchoolEuropeanHistoryTask(MMLUTask):
    data_name = "high_school_european_history"

@register_task("mmlu_high_school_geography")
class MMLU_HighSchoolGeographyTask(MMLUTask):
    data_name = "high_school_geography"

@register_task("mmlu_high_school_government_and_politics")
class MMLU_HighSchoolGovernmentAndPoliticsTask(MMLUTask):
    data_name = "high_school_government_and_politics"

@register_task("mmlu_high_school_macroeconomics")
class MMLU_HighSchoolMacroeconomicsTask(MMLUTask):
    data_name = "high_school_macroeconomics"

@register_task("mmlu_high_school_mathematics")
class MMLU_HighSchoolMathematicsTask(MMLUTask):
    data_name = "high_school_mathematics"

@register_task("mmlu_high_school_microeconomics")
class MMLU_HighSchoolMicroeconomicsTask(MMLUTask):
    data_name = "high_school_microeconomics"

@register_task("mmlu_high_school_physics")
class MMLU_HighSchoolPhysicsTask(MMLUTask):
    data_name = "high_school_physics"

@register_task("mmlu_high_school_psychology")
class MMLU_HighSchoolPsychologyTask(MMLUTask):
    data_name = "high_school_psychology"

@register_task("mmlu_high_school_statistics")
class MMLU_HighSchoolStatisticsTask(MMLUTask):
    data_name = "high_school_statistics"

@register_task("mmlu_high_school_us_history")
class MMLU_HighSchoolUSHistoryTask(MMLUTask):
    data_name = "high_school_us_history"

@register_task("mmlu_high_school_world_history")
class MMLU_HighSchoolWorldHistoryTask(MMLUTask):
    data_name = "high_school_world_history"

@register_task("mmlu_human_aging")
class MMLU_HumanAgingTask(MMLUTask):
    data_name = "human_aging"

@register_task("mmlu_human_sexuality")
class MMLU_HumanSexualityTask(MMLUTask):
    data_name = "human_sexuality"

@register_task("mmlu_international_law")
class MMLU_InternationalLawTask(MMLUTask):
    data_name = "international_law"

@register_task("mmlu_jurisprudence")
class MMLU_JurisprudenceTask(MMLUTask):
    data_name = "jurisprudence"

@register_task("mmlu_logical_fallacies")
class MMLU_LogicalFallaciesTask(MMLUTask):
    data_name = "logical_fallacies"

@register_task("mmlu_machine_learning")
class MMLU_MachineLearningTask(MMLUTask):
    data_name = "machine_learning"

@register_task("mmlu_management")
class MMLU_ManagementTask(MMLUTask):
    data_name = "management"

@register_task("mmlu_marketing")
class MMLU_MarketingTask(MMLUTask):
    data_name = "marketing"

@register_task("mmlu_medical_genetics")
class MMLU_MedicalGeneticsTask(MMLUTask):
    data_name = "medical_genetics"

@register_task("mmlu_miscellaneous")
class MMLU_MiscellaneousTask(MMLUTask):
    data_name = "miscellaneous"

@register_task("mmlu_moral_disputes")
class MMLU_MoralDisputesTask(MMLUTask):
    data_name = "moral_disputes"

@register_task("mmlu_moral_scenarios")
class MMLU_MoralScenariosTask(MMLUTask):
    data_name = "moral_scenarios"

@register_task("mmlu_nutrition")
class MMLU_NutritionTask(MMLUTask):
    data_name = "nutrition"

@register_task("mmlu_philosophy")
class MMLU_PhilosophyTask(MMLUTask):
    data_name = "philosophy"

@register_task("mmlu_prehistory")
class MMLU_PrehistoryTask(MMLUTask):
    data_name = "prehistory"

@register_task("mmlu_professional_accounting")
class MMLU_ProfessionalAccountingTask(MMLUTask):
    data_name = "professional_accounting"

@register_task("mmlu_professional_law")
class MMLU_ProfessionalLawTask(MMLUTask):
    data_name = "professional_law"

@register_task("mmlu_professional_medicine")
class MMLU_ProfessionalMedicineTask(MMLUTask):
    data_name = "professional_medicine"

@register_task("mmlu_professional_psychology")
class MMLU_ProfessionalPsychologyTask(MMLUTask):
    data_name = "professional_psychology"

@register_task("mmlu_public_relations")
class MMLU_PublicRelationsTask(MMLUTask):
    data_name = "public_relations"

@register_task("mmlu_security_studies")
class MMLU_SecurityStudiesTask(MMLUTask):
    data_name = "security_studies"

@register_task("mmlu_sociology")
class MMLU_SociologyTask(MMLUTask):
    data_name = "sociology"

@register_task("mmlu_us_foreign_policy")
class MMLU_USForeignPolicyTask(MMLUTask):
    data_name = "us_foreign_policy"

@register_task("mmlu_virology")
class MMLU_VirologyTask(MMLUTask):
    data_name = "virology"

@register_task("mmlu_world_religions")
class MMLU_WorldReligionsTask(MMLUTask):
    data_name = "world_religions"