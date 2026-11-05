#!/usr/bin/env python3
"""Annotate dashboard screenshots with professional callout arrows and explanatory labels."""

from PIL import Image, ImageDraw, ImageFont

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

def draw_callout(draw, target_xy, text_box_xy, badge_text, title_text, desc_text, color=(37, 99, 235), box_width=380):
    """Draws a target marker, leader line, and a stylish callout box."""
    font_badge = ImageFont.truetype(FONT_BOLD, 13)
    font_title = ImageFont.truetype(FONT_BOLD, 14)
    font_desc = ImageFont.truetype(FONT_REG, 12)
    
    tx, ty = target_xy
    bx, by = text_box_xy
    
    # 1. Target marker (pulsing circle)
    r = 6
    draw.ellipse([tx - r, ty - r, tx + r, ty + r], fill=color, outline=(255, 255, 255), width=2)
    draw.ellipse([tx - r - 4, ty - r - 4, tx + r + 4, ty + r + 4], outline=color, width=2)
    
    # 2. Text layout & wrap description
    words = desc_text.split()
    lines = []
    cur_line = []
    for w in words:
        cur_line.append(w)
        line_str = " ".join(cur_line)
        bbox = font_desc.getbbox(line_str)
        if bbox[2] - bbox[0] > (box_width - 24):
            cur_line.pop()
            lines.append(" ".join(cur_line))
            cur_line = [w]
    if cur_line:
        lines.append(" ".join(cur_line))
    
    box_height = 32 + len(lines) * 17 + 10
    box_rect = [bx, by, bx + box_width, by + box_height]
    
    # 3. Leader line
    if bx > tx:
        conn_x = bx
    elif bx + box_width < tx:
        conn_x = bx + box_width
    else:
        conn_x = bx + box_width // 2
        
    if by > ty:
        conn_y = by
    elif by + box_height < ty:
        conn_y = by + box_height
    else:
        conn_y = by + box_height // 2
        
    draw.line([(conn_x, conn_y), (tx, ty)], fill=color, width=3)
    
    # 4. Box background with shadow
    draw.rounded_rectangle([bx + 3, by + 3, bx + box_width + 3, by + box_height + 3], radius=7, fill=(0, 0, 0, 30))
    draw.rounded_rectangle(box_rect, radius=7, fill=(255, 255, 255, 248), outline=color, width=2)
    
    # 5. Badge pill
    badge_bbox = font_badge.getbbox(badge_text)
    bw = badge_bbox[2] - badge_bbox[0] + 14
    draw.rounded_rectangle([bx + 8, by + 7, bx + 8 + bw, by + 26], radius=4, fill=color)
    draw.text((bx + 15, by + 9), badge_text, font=font_badge, fill=(255, 255, 255))
    
    # 6. Title text
    draw.text((bx + 14 + bw + 8, by + 9), title_text, font=font_title, fill=(15, 23, 42))
    
    # 7. Description lines
    y_text = by + 32
    for line in lines:
        draw.text((bx + 10, y_text), line, font=font_desc, fill=(51, 65, 85))
        y_text += 17

def annotate_d5():
    # d5.png is (1500, 1800)
    im = Image.open("d5.png").convert("RGBA")
    overlay = Image.new("RGBA", im.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Callout 1: Triage banner (Target: 550, 370)
    draw_callout(draw, target_xy=(550, 370), text_box_xy=(950, 240),
                 badge_text="1", title_text="Assistive Follow-Up Triage",
                 desc_text="Assigns clinical priority: Tier 2 (Routine Preventive Tracking) automatically triggered by Grade 1 hepatic steatosis.",
                 color=(217, 119, 6), box_width=440)
    
    # Callout 2: De-identified text highlight (Target: 550, 650)
    draw_callout(draw, target_xy=(550, 650), text_box_xy=(950, 440),
                 badge_text="2", title_text="Verbatim Evidence Grounding",
                 desc_text="Report evidence is color-coded (Red = Affirmed Abnormality). Verified against text to reject ungrounded hallucinations.",
                 color=(220, 38, 38), box_width=440)
    
    # Callout 3: Sectionizer table (Target: 950, 650)
    draw_callout(draw, target_xy=(950, 650), text_box_xy=(950, 750),
                 badge_text="3", title_text="Anatomical Sectionizer",
                 desc_text="Partitions text into organ sub-sections (Liver, Gallbladder, Pancreas, etc.) and infers style format (Structured).",
                 color=(14, 165, 233), box_width=420)
                 
    # Callout 4: Structured Finding with RadLex & ICD-10 (Target: 650, 1380)
    draw_callout(draw, target_xy=(650, 1380), text_box_xy=(920, 1220),
                 badge_text="4", title_text="Ontology Codes (RadLex & ICD)",
                 desc_text="True Positive Extraction: Steatosis correctly affirmed with Grade 1 severity, mapped to RadLex RID39103 and ICD-10-CM K76.0.",
                 color=(37, 99, 235), box_width=460)
                 
    # Callout 5: ConText Negation Handling (Target: 650, 1435)
    draw_callout(draw, target_xy=(650, 1435), text_box_xy=(920, 1550),
                 badge_text="5", title_text="ConText Negation Assertion",
                 desc_text="Correctly identified 'No focal hepatic lesion' as Negated status with zero hallucinated abnormalities.",
                 color=(100, 116, 139), box_width=460)

    final = Image.alpha_composite(im, overlay).convert("RGB")
    final.save("d5_annotated.png", quality=95)
    print("d5_annotated.png created successfully!")

def annotate_page2():
    # page2.png is (1500, 1000)
    im = Image.open("page2.png").convert("RGBA")
    overlay = Image.new("RGBA", im.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Callout 1: Dynamic Filters (Target: 550, 175)
    draw_callout(draw, target_xy=(550, 175), text_box_xy=(960, 40),
                 badge_text="1", title_text="Dynamic Cohort Slicers",
                 desc_text="Filter population cohort dynamically by biological sex, age bracket, reporting style, and triage tier.",
                 color=(37, 99, 235), box_width=400)
                 
    # Callout 2: Disease KPIs (Target: 450, 260)
    draw_callout(draw, target_xy=(450, 260), text_box_xy=(30, 220),
                 badge_text="2", title_text="Workforce Disease KPIs",
                 desc_text="Prevalence metrics across workforce cohort (40% Hepatic Steatosis, 20% Cholelithiasis, 20% Renal Cysts).",
                 color=(14, 165, 233), box_width=330)
                 
    # Callout 3: Wilson CIs (Target: 1050, 520)
    draw_callout(draw, target_xy=(1050, 520), text_box_xy=(880, 580),
                 badge_text="3", title_text="Wilson 95% Confidence Intervals",
                 desc_text="Prevalence estimates calculated with rigorous Wilson score intervals, essential for small demographic sub-cohorts.",
                 color=(217, 119, 6), box_width=430)
                 
    # Callout 4: Follow-up tier distribution (Target: 1050, 880)
    draw_callout(draw, target_xy=(1050, 880), text_box_xy=(30, 780),
                 badge_text="4", title_text="Actionable Triage Distribution",
                 desc_text="Workforce stratification by follow-up urgency (Tier 1: Normal, Tier 2: Preventive Tracking, Tier 3: Clinician Review).",
                 color=(220, 38, 38), box_width=340)

    final = Image.alpha_composite(im, overlay).convert("RGB")
    final.save("page2_annotated.png", quality=95)
    print("page2_annotated.png saved successfully!")

def annotate_page3():
    # page3.png is (1500, 1000)
    im = Image.open("page3.png").convert("RGBA")
    overlay = Image.new("RGBA", im.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Callout 1: Primary Metrics (Target: 550, 310)
    draw_callout(draw, target_xy=(550, 310), text_box_xy=(30, 220),
                 badge_text="1", title_text="Bootstrap Micro F1 & Accuracy",
                 desc_text="Primary micro P/R/F1 evaluated with 1,000 cluster-bootstrap resamples over individual patient reports.",
                 color=(37, 99, 235), box_width=340)
                 
    # Callout 2: Concept vs Strict (Target: 550, 500)
    draw_callout(draw, target_xy=(550, 500), text_box_xy=(920, 420),
                 badge_text="2", title_text="Concept vs Strict Matching",
                 desc_text="Concept mode normalizes surface phrasing onto standard ontology concepts; strict mode measures exact string match.",
                 color=(14, 165, 233), box_width=420)
                 
    # Callout 3: Style Stratification (Target: 450, 760)
    draw_callout(draw, target_xy=(450, 760), text_box_xy=(30, 780),
                 badge_text="3", title_text="Style-Stratified Evaluation",
                 desc_text="Evaluation stratified across Structured, Telegraphic, and Unstructured reports with individual 95% bootstrap CIs.",
                 color=(16, 185, 129), box_width=340)

    final = Image.alpha_composite(im, overlay).convert("RGB")
    final.save("page3_annotated.png", quality=95)
    print("page3_annotated.png saved successfully!")

if __name__ == "__main__":
    annotate_d5()
    annotate_page2()
    annotate_page3()
