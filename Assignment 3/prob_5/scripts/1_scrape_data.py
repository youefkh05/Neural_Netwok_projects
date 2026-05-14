#!/usr/bin/env python3
"""
Arabic Text Scraper with Toxicity Labels
Fetches real Arabic text from multiple sources including HuggingFace toxicity datasets.
Creates a balanced dataset with toxic and non-toxic samples across MSA, Classical, and Dialects.
"""

import csv
import random
from pathlib import Path

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "raw_data.csv"

MSA_SENTENCES = [
    "تعتبر اللغة العربية من أقدم اللغات السامية وأكثرها انتشاراً في العالم",
    "يتحدث العربية أكثر من أربعمائة مليون نسمة حول العالم",
    "تتميز اللغة العربية بغناها بالمفردات والتراكيب البلاغية",
    "تحتوي اللغة العربية على ثمانية وعشرين حرفاً أصلياً",
    "تعد اللغة العربية لغة القرآن الكريم والحديث النبوي الشريف",
    "ترجع جذور اللغة العربية إلى أكثر من ألف وخمسمائة سنة",
    "تتميز الكتابة العربية باتصال حروفها وجمال خطها",
    "تفرعت اللغة العربية إلى العديد من اللهجات المحلية",
    "تستخدم اللغة العربية في العديد من الدول العربية والأفريقية",
    "تعد اللغة العربية إحدى اللغات الرسمية في الأمم المتحدة",
    "أثرت اللغة العربية في العديد من اللغات الأخرى",
    "تحتوي الموسوعة العربية على ملايين المقالات",
    "تتميز الموسيقى العربية بألحانها وإيقاعاتها المميزة",
    "تطورت الأدب العربي عبر العصور ليشمل الشعر والنثر",
    "يعتبر الشعر العربي من أقدم أنواع الأدب في التاريخ",
    "تتميز العمارة العربية بزخارفها الهندسية المعقدة",
    "انتشرت اللغة العربية مع انتشار الإسلام في العالم",
    "تختلف اللهجات العربية من منطقة لأخرى",
    "تعد القاهرة وبيروت ودمشق من أهم المراكز الثقافية",
    "تحظى اللغة العربية باهتمام كبير من المستشرقين",
    "الجامعات العربية تخرج الآلاف من الطلاب سنوياً",
    "الاقتصاد العربي يشهد نمواً مستمراً في الآونة الأخيرة",
    "الثقافة العربية غنية بالتراث والتاريخ العريق",
    "الفن العربي يتميز بألوانه الزاهية وتصاميمه الفريدة",
    "السياحة في الدول العربية تجذب ملايين الزوار",
    "الرياضة العربية تشهد تطوراً ملحوظاً في السنوات الأخيرة",
    "التعليم في الوطن العربي يمر بمراحل إصلاح مهمة",
    "البيئة العربية تتميز بتنوعها الجغرافي والمناخي",
    "الأسواق العربية تعج بالحركة والنشاط اليومي",
    "التكنولوجيا تغزو العالم العربي بوتيرة متسارعة",
]

CLASSICAL_SENTENCES = [
    "إنما الأعمال بالنيات وإنما لكل امرئ ما نوى",
    "لا يؤمن أحدكم حتى يحب لأخيه ما يحب لنفسه",
    "من كان يؤمن بالله واليوم الآخر فليقل خيراً أو ليصمت",
    "المسلم من سلم المسلمون من لسانه ويده",
    "لا ضرر ولا ضرار",
    "الدين النصيحة",
    "طلب العلم فريضة على كل مسلم",
    "خيركم من تعلم القرآن وعلمه",
    "من سلك طريقاً يلتمس فيه علماً سهل الله له به طريقاً إلى الجنة",
    "العلماء ورثة الأنبياء",
    "إن الله لا ينظر إلى صوركم وأموالكم ولكن ينظر إلى قلوبكم وأعمالكم",
    "من حسن إسلام المرء تركه ما لا يعنيه",
    "لا تغضب ولك الجنة",
    "تبسمك في وجه أخيك صدقة",
    "المرء مع من أحب",
    "إنما العلم بالتعلم والحلم بالتحلم",
    "من يرد الله به خيراً يفقهه في الدين",
    "استحيوا من الله حق الحياء",
    "من كذب علي متعمداً فليتبوأ مقعده من النار",
    "الطهور شطر الإيمان",
    "سباب المسلم فسوق وقتاله كفر",
    "من رأى منكم منكراً فليغيره بيده",
    "لا تحاسدوا ولا تباغضوا ولا تدابروا",
    "كونوا عباد الله إخواناً",
    "المؤمن للمؤمن كالبنيان يشد بعضه بعضاً",
    "من لا يرحم الناس لا يرحمه الله",
    "الراحمون يرحمهم الرحمن",
    "لا يدخل الجنة قاطع رحم",
    "إن الله جميل يحب الجمال",
    "إن الله يحب إذا عمل أحدكم عملاً أن يتقنه",
]

DIALECT_SENTENCES = {
    "egyptian": [
        "إيه الأخبار؟ كل حاجة تمام؟",
        "ما تشغلش بالك بالموضوع ده",
        "انا تعبان جداً النهاردة",
        "عامل إيه؟ من زمان عنك",
        "الجو حلو النهاردة مش كده؟",
        "عايز أروح السينما الليلة",
        "ما تقلقش كل حاجة هتبقى تمام",
        "إحنا هنروح الساحل الصيف ده",
        "الشارع مليان ناس دلوقتي",
        "أنا مش عارف أعمل إيه",
        "البيت كبير ومريح جداً",
        "الشغل تعبان بس لازم نكمل",
        "ما تعملش كده تاني",
        "الكلام ده مش صح خالص",
        "النهاردة يوم جميل جداً",
    ],
    "levantine": [
        "شلونك؟ شو أخبارك؟",
        "ما بعرف وين رايح",
        "هيدا المشروع مهم كتير",
        "بدروح عالمحطة هلق",
        "ما تعمل هيك تاني",
        "البيت عنا كبير وحلو",
        "شو رأيك نروح عالسوق",
        "ما بعرف شو أعمل",
        "الجو حلو اليوم مش هيك؟",
        "إمبارح كنت بالمدرسة",
        "بدنا نعمل حفلة بسيطة",
        "الشغل صعب بس لازم نكمل",
        "ما تخاف كل شي تمام",
        "عندي شغل كتير اليوم",
        "أنا تعبان اليوم",
        "الجو اليوم رائع كتير",
    ],
    "gulf": [
        "وش أخبارك؟ كيف حالك؟",
        "ما تخاف كل شي زين",
        "بغيت أروح السوق اليوم",
        "الجو حار اليوم وايد",
        "ما عندي وقت الحين",
        "أبي أروح البيت الحين",
        "الشغل وايد تعبان",
        "خلاص بنروح الحين",
        "ما تعبت من الشغل",
        "عندي وايد شغل اليوم",
        "البيت كبير ووايد حلو",
        "ما عندي أي فكرة",
        "بغيت أكل شي الحين",
        "السيارة وايد غالية",
        "خلاص خلاص ما تشيل هم",
        "الجو الليلة لطيف وايد",
    ],
    "maghrebi": [
        "كيفاش حالك؟",
        "واش كاين؟",
        "ما نعرف واش ندير",
        "بغيت نمشي للسوق",
        "الجو زوين اليوم",
        "واش عندك خدمة؟",
        "ما عندي والو",
        "حنا غادي نمشيو",
        "البيت كبير بزاف",
        "واش كاين جديد؟",
        "ما تخافش كلش مزيان",
        "عندي بزاف خدمة",
        "أنا تعيان اليوم",
        "الماكلة بنين بزاف",
        "واش راك؟",
        "الليل زوين بزاف",
    ],
}

TOXIC_KEYWORDS_MSA = [
    "غبي", "حقير", "خائن", "جبان", "كاذب", "سارق", "فاسد",
    "عميل", "خائن", "مجرم", "قتلة", "إرهابي",
]

TOXIC_KEYWORDS_DIALECT = {
    "egyptian": ["غبي", "تافه", "حرامي", "كذاب"],
    "levantine": ["غبي", "حقير", "حرامي", "كذاب"],
    "gulf": ["غبي", "تافه", "حرامي", "كذاب"],
    "maghrebi": ["غبي", "حقير", "حرامي", "كذاب"],
}

TOXIC_TEMPLATES_MSA = [
    "هذا الشخص {keyword} ولا يستحق الاحترام",
    "ماذا يقول هذا {keyword} في التلفزيون",
    "لا تصدقوا هذا {keyword} الكاذب",
    "هؤلاء مجموعة من {keyword} يجب محاسبتهم",
    "كيف يسمحون لهذا {keyword} بالتحدث",
    "هذا تصرف {keyword} ولا يليق بنا",
    "لا أثق في هذا {keyword} أبداً",
    "هذا {keyword} يدمر البلد",
]

TOXIC_TEMPLATES_DIALECT = {
    "egyptian": [
        "ده {keyword} عايز يضحك علينا",
        "ما تصدقوش ده {keyword}",
        "الراجل ده {keyword} وخاين",
        "مين ده {keyword} اللي بيكلم",
    ],
    "levantine": [
        "هيدا {keyword} وما بيستاهل",
        "ما تسمعوا لهيدا {keyword}",
        "هاد {keyword} وكتير خاين",
        "شو هيدا {keyword} اللي عم يحكي",
    ],
    "gulf": [
        "هذا {keyword} وايد كذاب",
        "ما تصدقون هذا {keyword}",
        "الشخص هذا {keyword} وخاين",
        "وش هذا {keyword} اللي يكلم",
    ],
    "maghrebi": [
        "هذا {keyword} كذاب بزاف",
        "ما تصدقش هذا {keyword}",
        "الشخص هذا {keyword} وخاين",
        "واش هذا {keyword} اللي كاين",
    ],
}


def load_huggingface_toxic_data(max_samples: int = 30):
    """Load Arabic toxic tweets from HuggingFace."""
    toxic_samples = []
    
    if not DATASETS_AVAILABLE:
        print("  HuggingFace datasets library not available")
        return toxic_samples
    
    dataset_options = [
        ("/cardiffnlp/tweet_sentiment_multilingual", "arabic", "text", "label"),
    ]
    
    for dataset_info in dataset_options:
        dataset_name = dataset_info[0]
        subset = dataset_info[1] if len(dataset_info) > 1 else None
        text_col = dataset_info[-2] if len(dataset_info) > 2 else "text"
        label_col = dataset_info[-1] if len(dataset_info) > 3 else "label"
        
        try:
            print(f"  Trying to load: {dataset_name}")
            if subset:
                ds = load_dataset(dataset_name, subset, split="train")
            else:
                ds = load_dataset(dataset_name, split="train")
            
            items = []
            
            for item in ds:
                text = str(item.get(text_col, "") or item.get("text", "") or "")
                if not text or len(text) < 10:
                    continue
                
                has_arabic = any('\u0600' <= c <= '\u06FF' for c in text)
                if not has_arabic:
                    continue
                
                items.append((text, "MSA", 0))
            
            random.shuffle(items)
            toxic_samples.extend(items[:max_samples])
            
            if toxic_samples:
                print(f"    Loaded {len(toxic_samples)} samples")
                break
                
        except Exception as e:
            print(f"    Failed: {e}")
            continue
    
    return toxic_samples[:max_samples]


def generate_synthetic_toxic_samples(count: int):
    """Generate synthetic toxic samples with Arabic content."""
    samples = []
    
    for _ in range(count):
        variety = random.choice(["MSA"] + list(DIALECT_SENTENCES.keys()))
        
        if variety == "MSA":
            keyword = random.choice(TOXIC_KEYWORDS_MSA)
            template = random.choice(TOXIC_TEMPLATES_MSA)
            text = template.format(keyword=keyword)
            samples.append((text, "MSA", 1))
        else:
            keyword = random.choice(TOXIC_KEYWORDS_DIALECT.get(variety, TOXIC_KEYWORDS_MSA))
            template = random.choice(TOXIC_TEMPLATES_DIALECT.get(variety, TOXIC_TEMPLATES_MSA))
            text = template.format(keyword=keyword)
            samples.append((text, variety, 1))
    
    return samples


def get_msa_sentences(count: int) -> list:
    sentences = [(s, "MSA", 0) for s in MSA_SENTENCES[:count]]
    return sentences


def get_classical_sentences(count: int) -> list:
    sentences = [(s, "Classical", 0) for s in CLASSICAL_SENTENCES[:count]]
    return sentences


def get_dialect_sentences(count: int) -> list:
    sentences = []
    dialects = list(DIALECT_SENTENCES.keys())
    per_dialect = count // len(dialects)
    
    for dialect, dialect_sentences in DIALECT_SENTENCES.items():
        selected = random.sample(dialect_sentences, min(per_dialect, len(dialect_sentences)))
        sentences.extend([(s, dialect, 0) for s in selected])
    
    random.shuffle(sentences)
    return sentences[:count]


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading toxic samples from HuggingFace...")
    hf_toxic = load_huggingface_toxic_data(max_samples=20)
    print(f"  Got {len(hf_toxic)} samples from HuggingFace")
    
    print("\nGenerating synthetic toxic samples...")
    synthetic_toxic_count = 25 - len(hf_toxic)
    synthetic_toxic = generate_synthetic_toxic_samples(synthetic_toxic_count)
    print(f"  Generated {len(synthetic_toxic)} synthetic toxic samples")
    
    total_toxic_target = 25
    all_toxic = (hf_toxic + synthetic_toxic)[:total_toxic_target]
    print(f"  Total toxic samples: {len(all_toxic)}")
    
    print("\nGetting non-toxic samples...")
    
    remaining = 100 - len(all_toxic)
    
    msa_count = remaining // 4
    classical_count = remaining // 4
    dialect_count = remaining - msa_count - classical_count
    
    non_toxic_samples = []
    non_toxic_samples.extend(get_msa_sentences(msa_count))
    non_toxic_samples.extend(get_classical_sentences(classical_count))
    non_toxic_samples.extend(get_dialect_sentences(dialect_count))
    
    print(f"  Target: {remaining} non-toxic, got {len(non_toxic_samples)}")
    
    while len(non_toxic_samples) < remaining:
        extra_needed = remaining - len(non_toxic_samples)
        non_toxic_samples.extend(get_msa_sentences(min(5, extra_needed)))
        if len(non_toxic_samples) < remaining:
            non_toxic_samples.extend(get_classical_sentences(min(5, remaining - len(non_toxic_samples))))
    
    print(f"  Final non-toxic count: {len(non_toxic_samples)}")
    
    print(f"  Got {len(non_toxic_samples)} non-toxic samples")
    
    all_data = all_toxic + non_toxic_samples
    random.shuffle(all_data)
    all_data = all_data[:100]
    
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "variety", "is_toxic"])
        for i, (text, variety, is_toxic) in enumerate(all_data, 1):
            writer.writerow([i, text, variety, is_toxic])
    
    print(f"\nSaved {len(all_data)} sentences to {OUTPUT_FILE}")
    
    print("\nDistribution by variety:")
    variety_counts = {}
    for _, variety, _ in all_data:
        variety_counts[variety] = variety_counts.get(variety, 0) + 1
    for variety, count in sorted(variety_counts.items()):
        print(f"  {variety}: {count}")
    
    print("\nDistribution by toxicity:")
    toxic_count = sum(1 for _, _, is_toxic in all_data if is_toxic)
    non_toxic_count = len(all_data) - toxic_count
    print(f"  Toxic: {toxic_count}")
    print(f"  Non-toxic: {non_toxic_count}")


if __name__ == "__main__":
    main()
