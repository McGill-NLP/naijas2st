"""Step 3 (few-shot): score-conditioned refinement with extra in-context examples.

Like ``step3_scored_translation.py`` but augments the refinement prompt
with few-shot demonstrations of (source, prior translation, score,
better translation) so Gemini learns the scoring/refinement pattern.
"""

from google import genai
from datasets import load_dataset, Audio
from huggingface_hub import login
import tempfile
import soundfile as sf
from itertools import islice
import os
import json

def get_fleurs_english_ref_for_one_sample(sample_id):
    """Fetch the English reference transcription for a FLEURS sample.

    Args:
        sample_id (int): FLEURS sample ID.

    Returns:
        str | None: The English transcription string, or ``None``.
    """
    ds_en = load_dataset("google/fleurs", "en_us", split="test", streaming=True)
    for sample in ds_en:
        if sample["id"] == sample_id:
            return sample.get("transcription")



def main():
    """Few-shot score-conditioned FLEURS retranslation with Gemini.

    Workflow:
        1. Authenticate with HuggingFace and instantiate a Gemini
           client.
        2. For each FLEURS language:
            - Stream the FLEURS test split for the LRL source.
            - Load the stage-2 results JSON containing the prior
              translation and its assigned score per ID.
            - Load ``few_shot_data/<code>/`` parallel wavs + ``.txt``
              files into refinement-style demonstrations (prior
              translation + score + corrected translation) so Gemini
              learns the scoring/refinement pattern.
            - For each test sample, write a temp WAV, build a
              multimodal prompt with the few-shot demos and the test
              audio + prior score, and call Gemini with retries.
            - Append ``{"id", "prediction", "reference", "source"}``
              and rewrite the per-language stage-3 JSON incrementally.

    Outputs:
        ``./RESULTS/stage3_few_shot/<code>.json`` per language.

    Returns:
        None.
    """
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    if HF_TOKEN:
        login(token=HF_TOKEN)
    else:
        print("Warning: No HF_TOKEN found. You may hit rate limits.")


    language_codes = {
        # "Irish":    "ga_ie",
        # "Welsh":     "cy_gb",
        # "Swahili":   "sw_ke",
        "Yoruba":    "yo_ng",
        # "Hausa":     "ha_ng",
        # "Igbo":      "ig_ng",
        # "Luganda":   "lg_ug",
    }


    # Make placeholder later
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

    client = genai.Client(api_key=GOOGLE_API_KEY)

    max_examples = None
    results_dir = "./RESULTS/stage3_few_shot_w_gt"
    os.makedirs(results_dir, exist_ok=True)

    for language_name, code in language_codes.items():
        print(f"\n→ Processing {language_name} ({code})…")
        scored_translation_dir = f"./RESULTS/stage2/{code}_with_scores_merged.json"

        # Load streaming test dataset
        ds_test = load_dataset("google/fleurs", code, split="test", streaming=True)
        ds_test = ds_test.cast_column("audio", Audio(sampling_rate=16_000))

        tmp_dir = tempfile.mkdtemp(prefix=f"fleurs_{code}_")

        out_path = os.path.join(results_dir, f"{code}.json")
        results = []

        # Iterate test split as a stream
        test_iter = ds_test if max_examples is None else islice(ds_test, max_examples)
        for sample in test_iter:
            print(f"  ↳ {sample['id']}")
            
            arr, sr = sample["audio"]["array"], sample["audio"]["sampling_rate"]
            file_name = f"{sample['id']}.wav"
            tmp_path = os.path.join(tmp_dir, f"test_{file_name}")
            sf.write(tmp_path, arr, sr)

            with open(scored_translation_dir) as fp:
                scored_translations = json.load(fp)
                for sentence in scored_translations:
                    if sentence["id"] == sample["id"]:
                        scored_prediction = sentence["prediction_original"]
                        score = sentence["score"]
                        reference = sentence["reference"]
                        source = sentence["source"]
                        print(f"    ↳ Found scored translation with score {score} for ID {sample['id']}")
                        break

            prompt_parts = [f"You are a translation expert. The following audio in {language_name} was previously machine translated to English and this machine translation was scored on its accuracy from [0] to [100]. Here are 5 examples of {language_name}, their ground truth English translation, their machine translation in English, accompanied by a translation score (0-100) given by a native {language_name} speaker. Given the audio, the scored examples and the machine translation and its given score, transcribe the audio from {language_name} and use the transcription to translate it to English accurately, avoiding the mistakes made in the previous machine translation according to the score. Provide only the new translation, not the transcription or any additional text or formatting. \n"]

            test_audio = client.files.upload(file=tmp_path)
            prompt_parts.append(test_audio)

            prompt_parts.append("\nHere are the 5 scored examples:\n")
            prompt_parts.append(f"Example 1: {language_name} sentence: 'Àwọn onímọ̀ ìjìnlẹ̀ sáyẹ̀nsì láti ilé ìkẹ́ẹ̀kọ́ gíga ti ìsègun Stanford lọ́jọ́ ajé ti kéde ìdásílẹ̀ irinṣẹ́ ìwádìí tuntun tí ó le tó nǹkan lẹ́sẹẹsẹ pẹ̀lú bí wọ́n bá se rí: irinṣẹ́ kéreké tí a lè tẹ̀ jáde pẹ̀lú lílo irinṣẹ́ ìtẹ̀wé ìgbàlódé pẹ̀lú owó ẹyọ fún ọ̀kọ̀ọ̀kan.' \n Ground Truth: 'On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool that can sort cells by type: a tiny printable chip that can be manufactured using standard inkjet printers for possibly about one U.S. cent each.' \n Machine translation: 'Scientists from Stanford University's Earthquake Institute on Economic Day have announced the launch of a new research tool that's roughly the same size as it ever was: a small tool that can be printed using a modern printing press with a penny each.'\n Score: '61'\n")
            prompt_parts.append(f"Example 2: {language_name} sentence: 'Olùwádìí àgbà sọ pé èyí le è tètè mú kí wọn tètè mọ nípa jẹjẹrẹ, ikọ́ fee HIV, àti ibà lára àwọn aláàrẹ̀ ni àwọn orílẹ̀ èdè tí kòní owó púpọ̀, ní ibi tí àwọn tí wọ́n ń borí àìsàn bíì jẹjẹrẹ ọyàn jẹ bí ìlàjì àwọn olówó ní ìlú náà.'\n Ground Truth: 'Lead researchers say this may bring early detection of cancer, tuberculosis, HIV and malaria to patients in low-income countries, where the survival rates for illnesses such as breast cancer can be half those of richer countries.' \n Machine translation: 'The lead researcher said this could lead to speedy awareness of cancer, HIV infection, and malaria among presidents in low-income countries, where the number of cancer survivors is as high as half of the city's wealthy.'\n Score: '83'\n")
            prompt_parts.append(f"Example 3: {language_name} sentence: 'Ilé isé oníròyìn abẹ́lé́ fi síta wípé ọkọ̀ panápaná ilé isé ọkọ̀ òfuurufú subú nígbà tí o ń sisẹ́.'\n Ground Truth: ' Local media reports an airport fire vehicle rolled over while responding.' \n Machine translation: 'A local news agency reported that the airport fire truck crashed while you were preparing.'\n Score: '66'\n")
            prompt_parts.append(f"Example 4: {language_name} sentence: 'Fidali, omo odun-28 ti darapọ̀ mọ́ ẹgbẹ́ agbáboolu Basilona ní àkókò ìdíjé méta sẹ́yin, láti Sefila.'\n Ground Truth: '28-year-old Vidal had joined Barça three seasons ago, from Sevilla.'\n Machine translation: 'Fidali, 28, joined Basilona's football team three seasons ago from Sefila.'\n Score: '100'\n")
            prompt_parts.append(f"Example 5: {language_name} sentence: 'Ní aago mọ́kànlá kọjá ogún ìsẹ́jù, iléeṣẹ́ ọlọ́ọ̀pá tí ní káwọn olùsèfẹ̀hónúhàn sún sẹ́yìn, sọ pé wọ́n nílò láti sètò ìfẹ̀hónúhàn pẹ̀lú súnkẹrẹ fàkẹrẹ tó ń kó jọ.'\n Ground Truth: 'At 11:20, the police asked the protesters to move back on to the pavement, stating that they needed to balance the right to protest with the traffic building up.'\n Machine translation: 'At 11 o'clock in the afternoon, the news agency, which ordered the protesters to stand back, called for an immediate ceasefire.'\n Score: '25'\n")

            prompt_parts.append(f"\nPrevious machine translation: {scored_prediction}\n")
            prompt_parts.append(f"Score of previous machine translation: {score}\n")
            prompt_parts.append(f"Please provide the new accurate English translation based on the audio, its transcription and the previously scored machine translation. Provide only the new translation, without any additional text or formatting.\n")


            success = False
            number_of_retries = 0
            while not success:
                print(f"    ↳ Attempt {number_of_retries + 1}")
                number_of_retries += 1
                if number_of_retries > 5:
                    print(f"  ↳ failed to process, moving on...")
                    break
                try:
                    resp = client.models.generate_content(
                                model="gemini-2.5-flash",
                                contents=prompt_parts
                            )

                    results.append({
                                "id": sample["id"],
                                "prediction": resp.text.strip(),
                                "reference": reference,
                                "source": source
                            })
                    success = True
                except Exception as e:
                    success = False
                    print(f"caught error, retrying: {e}")

        # Save results per language
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(results, fp, ensure_ascii=False, indent=2)

        print(f"  ↳ saved {len(results)} translations to {out_path}")


if __name__ == "__main__":
    main()
