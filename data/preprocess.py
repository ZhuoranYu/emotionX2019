import os
import json

def bert_format():
    dataset = ['Friends', 'EmotionPush']

    for data in dataset:
        with open(os.path.join('./', data, data.lower() + '.augmented.json')) as fp:
            all_dialogues = json.loads(fp.read())

        dialogue_id = 0
        dialogueList = []
        line_index = 0
        for dialogue in all_dialogues:
            utt_id = 0
            uttList = []
            for utt_obj in dialogue:
                nested_id = "d" + str(dialogue_id).zfill(4) + '_' + 'u' + str(utt_id).zfill(4)
                utt_en = utt_obj['utterance']
                utt_de = utt_obj['utterance_de']
                utt_fr = utt_obj['utterance_fr']
                utt_it = utt_obj['utterance_it']

                with open(os.path.join('./', data + '_bert', data.lower() + '_en.txt'), 'a+') as fp:
                    fp.write(utt_en + '\n')

                with open(os.path.join('./', data + '_bert', data.lower() + '_de.txt'), 'a+') as fp:
                    fp.write(utt_de + '\n')

                with open(os.path.join('./', data + '_bert', data.lower() + '_fr.txt'), 'a+') as fp:
                    fp.write(utt_fr + '\n')

                with open(os.path.join('./', data + '_bert', data.lower() + '_it.txt'), 'a+') as fp:
                    fp.write(utt_it + '\n')

                uttDict = {'uid': nested_id, 'annotation': utt_obj['annotation'], 'line_index': line_index}
                uttList.append(uttDict)

                # updates
                line_index += 1
                utt_id += 1
            with open(os.path.join('./', data + '_bert', data.lower() + '_en.txt'), 'a+') as fp:
                fp.write('\n')

            with open(os.path.join('./', data + '_bert', data.lower() + '_de.txt'), 'a+') as fp:
                fp.write('\n')

            with open(os.path.join('./', data + '_bert', data.lower() + '_fr.txt'), 'a+') as fp:
                fp.write('\n')

            with open(os.path.join('./', data + '_bert', data.lower() + '_it.txt'), 'a+') as fp:
                fp.write('\n')
            dialogueList.append(uttList)
            line_index += 1
            dialogue_id += 1

        with open(os.path.join('./', data, 'processed.json'), 'w') as fp:
            json.dump(dialogueList, fp)






if __name__ == "__main__":
    bert_format()