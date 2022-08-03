import streamlit as st
import pandas as pd
import numpy as np

st.title("Seach by meaning of the text")

st.header("Start of the Text Input Section")
q = st.text_input("พิมพ์ข้อความที่ต้องการค้นหา") #text
if st.button("กดค้นหา"):
    st.write("ค้นหาประโยคหรือคำว่า", q)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
        
    #read_from_file()
    a1 = "แนวทางประชาธิปไตยที่เหมาะสมในทัศนะผู้เขียน “ประชาธิปไตยไม่ใช่รูปแบบการปกครองที่ดีที่สุดแต่เป็นรูปแบบการปกครองที่เลวน้อยที่สุด” คำกล่าวข้างต้น จึงอาจเป็นคำตอบของการที่ ประชาธิปไตยเป็นระบอบการเมืองการปกครองที่โลกปัจจุบันนิยมใช้มากที่สุด โดยยึดหลักปฏิบัติพื้นฐานคล้ายกันคือมีโครงสร้างที่บัญญัติไว้ในรัฐธรรมนูญ ประชาชนเลือกผู้แทนเข้าไปทำหน้าที่นิติบัญญัติในรัฐสภาและเลือกผู้แทนไปเป็นรัฐบาลในฝ่ายบริหารอีกต่อหนึ่ง"
        
    b2 ="บทที่ 1 ประชาธิปไตยในบริบทสากล"
    b3 = "1.1 ประชาธิปไตย คืออะไร คำว่า ประชาธิปไตย ประกอบด้วยคำว่า ประชา หมายถึงหมู่คนคือปวงชนกับคำว่า อธิปไตย หมายถึงความเป็นใหญ่ตังนั้นคำว่าประชาธิปไตยจึงหมายถึงความเป็นใหญ่ของปวงชน ตามคำกล่าวของอดีตประธานาธิบดีแห่งสหรัฐอเมริกาอับราฮัมลินคอล์น ประชาธิปไตยคือการปกครองของประชาชนโดยประชาชนและเพื่อประชาชน "
        
    b4 = "1.2 รูปแบบการปรองระบอบประชาธิปไตย 1.2.1 ประชาธิปไตยทางตรง (Direct Democracy) 1.2.2  ประชาธิปไตยแบบมีผู้แทน (Representative Democracy)"

    b5 = " 2.1 ประชาธิปไตยทางตรง ประชาธิปไตยทางตรงหรือประชาธิปไตยบริสุทธิ์คือรูปแบบการปกครองโดยที่พลเมืองสามารถมีส่วนร่วมกับการตัดสินใจใดๆได้โดยตรงโดยไม่ต้องอาศัยคนกลางหรือผู้ทำหน้าที่แทนตน รวมไป ถึงการร่างกฎหมายและนโยบายของรัฐบาลเช่น ประชาธิปไตยแบบเอเธนส์ "
    
    b6 =  "2.2 ประชาธิปไตยแบบมีผู้แทน ประชาธิปไตยแบบมีผู้แทนหรือประชาธิปไตยทางอ้อมคือการปกครองที่ประชาชนเลือกผู้แทนไปทำหน้าที่ร่วมตัดสินใจทางการเมืองแทนตนในรัฐสภาซึ่งผู้แทนอาจมาจากการเลือกตั้ง หรือมาจากการสรรหาตามรัฐธรรมนูญหรือมาจากการกำหนดโดยพรรคการเมืองหรืออาจใช้รูปแบบผสมผสานกันรูปแบบที่ใช้กันอย่างแพร่หลายที่สุดในสมัยปัจจุบันแบ่งเป็น"
        
    c7 =  "บทที่ 2 ประชาธิปไตยในประเทศไทย 2.1 รูปแบบการปรองระบอบประชาธิปไตย "
    c8 = "2.1.1 คิดว่าประชาธิปไตยคือการเลือกตั้ง ให้ความสำคัญกับวิธีการแทนการปลูกฝังเรียนรู้วิถีประชาธิปไตย"
    c9 = "2.1.2 คิดว่ารัฐธรรมนูญดีจะทำให้เกิดประชาธิปไตยที่เข็มแข็ง จึงทำให้เน้นการแก้ไขตัวบทแทนการแก้ที่รากของปัญหาคือ คนการเปลี่ยนแปลงรัฐธรรมนูญ บ่อยครั้งจนเกินไปทำให้ขาดความต่อเนื่องในการบังคับใช้"
        
    dd = [a1,b2,b3,b4, b5,b6,c7,c8,c9,q]


    vectorizer = TfidfVectorizer()
    trsf=vectorizer.fit_transform(dd)
    pd.DataFrame(trsf.toarray(),columns = vectorizer.get_feature_names(),index=['a1','b2','b3','b4','b5','b6','c7','c8','c9','q'])

        #from sklearn.metrics.pairwise import cosine_similarity
        #cosine_similarity(trsf[0], trsf)

    doca1 = trsf[0:1].todense()
    docb2 = trsf[1:2].todense()
    docb3 = trsf[2:3].todense()
    docb4 = trsf[3:4].todense()
    docb5 = trsf[4:5].todense()
    docb6 = trsf[5:6].todense()
    docc7 = trsf[6:7].todense()
    docc8 = trsf[7:8].todense()
    docc9 = trsf[8:9].todense()

    docq = trsf[9:10].todense()


    doca1.shape,docb2.shape,docb3.shape,docb4.shape,docb5.shape,docb6.shape,docc7.shape,docc8.shape,docc9.shape,docq.shape
    doc_a1 = []
    for i in range(len(doca1[0])):
        doc_a1.append(doca1[i])


    doc_b2 = []
    for i in range(len(docb2[0])):
        doc_b2.append(docb2[i])


    doc_b3 = []
    for i in range(len(docb3[0])):
        doc_b3.append(docb3[i])


    doc_b4 = []
    for i in range(len(docb4[0])):
         doc_b4.append(docb4[i])


    doc_b5 = []
    for i in range(len(docb5[0])):
        doc_b5.append(docb5[i])


    doc_b6 = []
    for i in range(len(docb6[0])):
        doc_b6.append(docb6[i])


    doc_c7 = []
    for i in range(len(docc7[0])):
        doc_c7.append(docc7[i])


    doc_c8 = []
    for i in range(len(docc8[0])):
        doc_c8.append(docc8[i])


    doc_c9 = []
    for i in range(len(docc9[0])):
        doc_c9.append(docc9[i])

    doc_q = []
    for i in range(len(docq[0])):
        doc_q.append(docq[i])


    #from scipy.spatial import distance

    doc_a1 = np.squeeze(np.asarray(doc_a1))
    doc_b2 = np.squeeze(np.asarray(doc_b2))
    doc_b3 = np.squeeze(np.asarray(doc_b3))
    doc_b4 = np.squeeze(np.asarray(doc_b4))
    doc_b5 = np.squeeze(np.asarray(doc_b5))
    doc_b6 = np.squeeze(np.asarray(doc_b6))
    doc_c7 = np.squeeze(np.asarray(doc_c7))
    doc_c8 = np.squeeze(np.asarray(doc_c8))
    doc_c9 = np.squeeze(np.asarray(doc_c9))
    doc_q = np.squeeze(np.asarray(doc_q))

    dot = sum(a1*b1 for a1,b1 in zip(doc_q, doc_a1))
    norm_a1 = sum(a1*a1 for a1 in doc_q)** 0.5
    norm_b1 = sum(b1*b1 for b1 in doc_a1) ** 0.5
    co1 = dot / (norm_a1*norm_b1)

    dot = sum(a2*b2 for a2,b2 in zip(doc_q, doc_b2))
    norm_a2 = sum(a2*a2 for a2 in doc_q)** 0.5
    norm_b2 = sum(b2*b2 for b2 in doc_b2) ** 0.5
    co2 = dot / (norm_a2*norm_b2)
        
    dot = sum(a3*b3 for a3,b3 in zip(doc_q, doc_b3))
    norm_a3 = sum(a3*a3 for a3 in doc_q)** 0.5
    norm_b3 = sum(b3*b3 for b3 in doc_b3) ** 0.5
    co3 = dot / (norm_a3*norm_b3)

    dot = sum(a4*b4 for a4,b4 in zip(doc_q, doc_b4))
    norm_a4 = sum(a4*a4 for a4 in doc_q)** 0.5
    norm_b4 = sum(b4*b4 for b4 in doc_b4) ** 0.5
    co4 = dot / (norm_a4*norm_b4)


    dot = sum(a5*b5 for a5,b5 in zip(doc_q, doc_b5))
    norm_a5 = sum(a5*a5 for a5 in doc_q)** 0.5
    norm_b5 = sum(b5*b5 for b5 in doc_b5) ** 0.5
    co5 = dot / (norm_a5*norm_b5)

    dot = sum(a6*b6 for a6,b6 in zip(doc_q, doc_b6))
    norm_a6 = sum(a6*a6 for a6 in doc_q)** 0.5
    norm_b6 = sum(b6*b6 for b6 in doc_b6) ** 0.5
    co6 = dot / (norm_a6*norm_b6)

    dot = sum(a7*b7 for a7,b7 in zip(doc_q, doc_c7))
    norm_a7 = sum(a7*a7 for a7 in doc_q)** 0.5
    norm_b7 = sum(b7*b7 for b7 in doc_c7) ** 0.5
    co7 = dot / (norm_a7*norm_b7)

    dot = sum(a8*b8 for a8,b8 in zip(doc_q, doc_c8))
    norm_a8 = sum(a8*a8 for a8 in doc_q)** 0.5
    norm_b8 = sum(b8*b8 for b8 in doc_c8) ** 0.5
    co8 = dot / (norm_a8*norm_b8)

    dot = sum(a9*b9 for a9,b9 in zip(doc_q, doc_c9))
    norm_a9 = sum(a9*a9 for a9 in doc_q)** 0.5
    norm_b9 = sum(b9*b9 for b9 in doc_c9) ** 0.5
    co9 = dot / (norm_a9*norm_b9)

    dict_s = {
            #'id':['a1','b2','b3','b4','b5','b6','c7','c8','c9'],
            'บท':['บทนำ','บทที่1','บทที่1','บทที่1','บทที่1','บทที่1','บทที่2','บทที่2','บทที่2'],
            'หัวข้อย่อย': ['-','1*','1.1','1.2','2.1','2.2','2*','2.1.1','2.1.2'],
            'ข้อความ':["แนวทางประชาธิปไตยที่เหมาะสมในทัศนะผู้เขียน “ประชาธิปไตยไม่ใช่รูปแบบการปกครองที่ดีที่สุดแต่เป็นรูปแบบการปกครองที่เลวน้อยที่สุด” คำกล่าวข้างต้น จึงอาจเป็นคำตอบของการที่ ประชาธิปไตยเป็นระบอบการเมืองการปกครองที่โลกปัจจุบันนิยมใช้มากที่สุด โดยยึดหลักปฏิบัติพื้นฐานคล้ายกันคือมีโครงสร้างที่บัญญัติไว้ในรัฐธรรมนูญ ประชาชนเลือกผู้แทนเข้าไปทำหน้าที่นิติบัญญัติในรัฐสภาและเลือกผู้แทนไปเป็นรัฐบาลในฝ่ายบริหารอีกต่อหนึ่ง",
            "บทที่ 1 ประชาธิปไตยในบริบทสากล",
            "1.1 ประชาธิปไตย คืออะไร คำว่า ประชาธิปไตย ประกอบด้วยคำว่า ประชา หมายถึงหมู่คนคือปวงชนกับคำว่า อธิปไตย หมายถึงความเป็นใหญ่ตังนั้นคำว่าประชาธิปไตยจึงหมายถึงความเป็นใหญ่ของปวงชน ตามคำกล่าวของอดีตประธานาธิบดีแห่งสหรัฐอเมริกาอับราฮัมลินคอล์น ประชาธิปไตยคือการปกครองของประชาชนโดยประชาชนและเพื่อประชาชน ",
            "1.2 รูปแบบการปรองระบอบประชาธิปไตย 1.2.1 ประชาธิปไตยทางตรง (Direct Democracy) 1.2.2  ประชาธิปไตยแบบมีผู้แทน (Representative Democracy)",
            "2.1 ประชาธิปไตยทางตรง ประชาธิปไตยทางตรงหรือประชาธิปไตยบริสุทธิ์คือรูปแบบการปกครองโดยที่พลเมืองสามารถมีส่วนร่วมกับการตัดสินใจใดๆได้โดยตรงโดยไม่ต้องอาศัยคนกลางหรือผู้ทำหน้าที่แทนตน รวมไป ถึงการร่างกฎหมายและนโยบายของรัฐบาลเช่น ประชาธิปไตยแบบเอเธนส์ ",
            "2.2 ประชาธิปไตยแบบมีผู้แทน ประชาธิปไตยแบบมีผู้แทนหรือประชาธิปไตยทางอ้อมคือการปกครองที่ประชาชนเลือกผู้แทนไปทำหน้าที่ร่วมตัดสินใจทางการเมืองแทนตนในรัฐสภาซึ่งผู้แทนอาจมาจากการเลือกตั้ง หรือมาจากการสรรหาตามรัฐธรรมนูญหรือมาจากการกำหนดโดยพรรคการเมืองหรืออาจใช้รูปแบบผสมผสานกันรูปแบบที่ใช้กันอย่างแพร่หลายที่สุดในสมัยปัจจุบันแบ่งเป็น",
            "บทที่ 2 ประชาธิปไตยในประเทศไทย 2.1 รูปแบบการปรองระบอบประชาธิปไตย ",
            "2.1.1 คิดว่าประชาธิปไตยคือการเลือกตั้ง ให้ความสำคัญกับวิธีการแทนการปลูกฝังเรียนรู้วิถีประชาธิปไตย",
            "2.1.2 คิดว่ารัฐธรรมนูญดีจะทำให้เกิดประชาธิปไตยที่เข็มแข็ง จึงทำให้เน้นการแก้ไขตัวบทแทนการแก้ที่รากของปัญหาคือ คนการเปลี่ยนแปลงรัฐธรรมนูญ บ่อยครั้งจนเกินไปทำให้ขาดความต่อเนื่องในการบังคับใช้"],
            'ความคล้าย':[str(co1),str(co2),str(co3),str(co4),str(co5),str(co6),str(co7),str(co8),str(co9)]} 

        

    df = pd.DataFrame(dict_s)
    st.write(df)
        
       