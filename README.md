# Keyword-Extraction-dosilt

한국어 문서에서 Unsupervised 기반으로 키워드를 추출하는 기법들에 대해 공부/연구 자료입니다.  

**1.** TextRank와 tokenizer를 결합한 키워드 추출 방법  
**2.** SIFRank와 비슷한 방식이지만 KoBERT를 이용 및 EmbedRank와 비슷하게 구현한 키워드 추출 방법(KoBERT)  
**3.** KeyBERT를 KoBERT를 활용하여 한국어에 대해서 키워드 추출 방법(KoSBERT+NLI)  
**4.** DP를 이용하여 keyword/keyphrase를 사전 추출하지 않고 추출 방법 (연구중, KoBERT)  





  
<table>
  <tr>
    <td>   입력 문장   </td>
    <td colspan="10"> 텍스트 요약은 텍스트의 관련 정보를 나타내는 여러 가지 방법으로 구성된 광범위한 항목입니다. 이 설명서에 설명된 문서 요약 기능을 통해 추출 텍스트 요약을 사용하여 문서의 요약을 생성할 수 있습니다. 원본 콘텐츠 내에서 가장 중요하거나 관련성 있는 정보를 집합적으로 나타내는 문장을 추출합니다. 이 기능은 너무 길어서 읽을 수 없다고 생각할 수 있는 콘텐츠를 줄이도록 설계되었습니다. 예를 들어 문서, 논문 또는 문서를 주요 문장으로 압축할 수 있습니다. </td>
  </tr>
  <tr>
    <td>   <b>1</b> + space   </td>
    <td> 수 </td>
    <td> 이 </td>
    <td> 정보를 </td>
    <td> 나타내는 </td>
    <td> 있는 </td>
    <td> 요약을 </td>
    <td> 텍스트 </td>
    <td> 있습니다. </td>
    <td> 논문 </td>
    <td> 문서 </td>
  </tr>
  
  <tr>
    <td>   <b>1</b> + Komoran.morphs   </td>
    <td> 문서 </td>
    <td> 요약 </td>
    <td> 수 </td>
    <td> 텍스트 </td>
    <td> 정보 </td>
    <td> 콘텐츠 </td>
    <td> 기능 </td>
    <td> 추출 </td>
    <td> 문장 </td>
    <td> 광범위 </td>
  </tr>
  
  <tr>
    <td>   <b>1</b> + Okt.phrase   </td>
    <td> 방법 </td>
    <td> 텍스트의 관련 정보 </td>
    <td> 여러 </td>
    <td> 가지 </td>
    <td> 정보 </td>
    <td> 여러 가지 </td>
    <td> 여러 가지 방법 </td>
    <td> 관련 </td>
    <td> 텍스트의 관련 </td>
    <td> 항목 </td>
  </tr>
  
  <tr>
    <td>   <b>2</b> + Komoran + diversity=0   </td>
    <td> 압축 </td>
    <td> 추출 </td>
    <td> 원본 </td>
    <td> 항목 </td>
    <td> 텍스트 </td>
    <td> 콘텐츠 </td>
    <td> 생성 </td>
    <td> 요약 </td>
    <td> 집합 </td>
    <td> 문장 </td>
  </tr>
  
  <tr>
    <td>   <b>2</b> + Komoran + diversity=0.5   </td>
    <td> 압축 </td>
    <td> 설명서 </td>
    <td> 수 </td>
    <td> 텍스트 </td>
    <td> 원본 </td>
    <td> 집합 </td>
    <td> 항목 </td>
    <td> 논문 </td>
    <td> 기능 </td>
    <td> 관련 </td>
  </tr>
  
  <tr>
    <td>   <b>3</b> + Komoran   </td>
    <td> 설명 </td>
    <td> 논문 </td>
    <td> 정보 </td>
    <td> 압축 </td>
    <td> 콘텐츠 </td>
    <td> 항목 </td>
    <td> 설명서 </td>
    <td> 텍스트 </td>
    <td> 요약 </td>
    <td> 문서 </td>
  </tr>
  
  <tr>
    <td>   <b>4</b> + space   </td>
    <td> 압축할 </td>
    <td> 항목입니다 </td>
    <td> 추출 </td>
    <td> 원본 </td>
    <td> 있습니다 </td>
    <td> 요약을 </td>
    <td> 추출합니다 </td>
    <td> 문장으로 </td>
    <td> 요약은 </td>
    <td> 주요 </td>
  </tr>  
  
  <tr>
    <td>   <b>4</b> + DP   </td>
    <td colspan="3"> 문서를 주요 문장으로 압축할 수 있습니다 </td>
    <td colspan="4"> 통해 추출 텍스트 요약을 사용하여 문서의 요약을 </td>
    <td colspan="3"> 집합적으로 나타내는 문장을 추출합니다 </td>
  </tr>
</table>
