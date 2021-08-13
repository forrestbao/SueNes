import xml2dict, json, os
from summ_eval.bleu_metric import BleuMetric
from summ_eval.cider_metric import CiderMetric
from summ_eval.s3_metric import S3Metric    # Use sklearn 0.21.X
from summ_eval.meteor_metric import MeteorMetric, enc
from summ_eval.bert_score_metric import BertScoreMetric
# from summ_eval.rouge_metric import RougeMetric
# rouge = RougeMetric(rouge_args="-n 4 -w 1.2 -m  -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -a")
# rouge = RougeMetric(rouge_args="-c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a")
# rouge = RougeMetric()
# rouge_dict = bertscore.evaluate_example(hyp, refs)
#hyp = "ARVADA, Colo.\n(AP) -- Columbine High School junior Matt Wells took time out from the grieving world around him to crack a smile, chew on a wad of sunflower seeds and talk a little baseball.\n``It's just good to be smiling and laughing,'' he said.\nWells, a 16-year-old catcher on Columbine's varsity baseball team, watched the junior varsity play Arvada West High School on Wednesday.\nThere were cheers and grins here, eight days after teen-age gunmen walked the halls of their school and killed 12 students and a teacher before killing themselves.\n"
#refs = ['On 20 April 1999, a massacre occurred at Columbine High School in Littleton, Colorado, a suburb of Denver.\nTwelve students, one teacher, and the two perpetrators, Eric Harris and Dylan Klebold, who planned the massacre over a two year period, were killed.\nThe perpetrators, who died from self inflicted wounds, were considered disaffected outcasts, frequently abused by school athletes.\nBecause of the uneasiness of students, their classes were moved to the neighboring school Chatfield.\nVice President Gore addressed 70,000 attendees at a Memorial Service and Republican presidential candidate Pat Buchanan cited the incident as rationale for stricter gun laws.\n', 'In the worst school killing in U.S. history, two students at Columbine High School in Littleton, Colorado, a Denver suburb, entered their school on Tuesday, April 20, 1999, to shoot and bomb.\nAt the end 15 were dead and dozens injured.\nThe dead included the two students, Eric Harris and Dylan Klebold, who killed themselves.\nHarris and Klebold were enraged by what they considered taunts and insults from classmates and had planned the massacre for more than a year.\nThe school is a sealed crime scene and Columbine students will complete the school year at a nearby high school.\n', "In Littleton, Colorado on Tuesday morning, April 20, 1999, Columbine High School students Eric Harris and Dylan Klebold, wearing black trench coats, entered their school with guns and pipe bombs.\nThey killed 12 students and teacher Dave Sanders before killing themselves.\nThey planned the massacre for a year, enraged by classmates' taunts and insults.\nMany victims were in the cafeteria.\nInvestigators believed the gunmen had help from others.\nColumbine remained closed and students were to finish the school year at nearby Chatfield High.\nCopycat threats occurred nationwide.\nThe world grieved with Columbine.\nVP Al Gore spoke at the memorial.\n", "The worst school killing in US history occurred Tuesday 20 April at Columbine High School in a Denver suburb.\nTwo students, enraged at treatment by schoolmates, killed 12 students and one teacher and wounded dozens before killing themselves.\nThe entire nation mourned the massacre.\nOn Sunday Vice President Gore and the Colorado governor addressed an outdoor memorial service attended by 70,000.\nThe school was closed and turned into a crime scene, as police investigated whether others were involved.\nClasses were scheduled to resume at a neighboring school, which the school's 1965 students would share for the remainder of the year.\n"]

def calc_one(hyp, refs, scorers):
    score = {}
    for scorer in scorers:
        score.update(scorer.evaluate_example(hyp, refs))

    return score

def main():
    scorers = [BleuMetric(), CiderMetric(), S3Metric(), MeteorMetric(), BertScoreMetric()]

    eval_path = '/home/nkwbtb/Downloads/TAC2010/GuidedSumm2010_eval/ROUGE/'
    in_file = 'rouge_A.in'

    xml = ''
    with open(os.path.join(eval_path, in_file), 'r', encoding='utf-8') as f:
        xml = f.read()

    eval = xml2dict.parse(xml)
    eval = eval["ROUGE_EVAL"]["EVAL"]

    results = {}
    for exp in eval:
        print(exp["@ID"])
        MODELS = exp["MODELS"]["M"]
        PEERS = exp["PEERS"]["P"]
        
        refs = []
        for model in MODELS:
            model_file = model["#text"]
            with open(os.path.join(eval_path, 'models', model_file), 'r', encoding='cp1252') as f:
                refs.append(f.read())
        
        for peer in PEERS:
            peer_file = peer["#text"]
            print(peer_file)
            peer_text = ''
            with open(os.path.join(eval_path, 'peers', peer_file), 'r', encoding='cp1252') as f:
                peer_text = f.read()

            results[peer_file] = calc_one(peer_text, refs, scorers)
    
    with open('baselines.json', 'w', encoding='utf-8') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()
