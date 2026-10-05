"""Build the research report from saved experiment evidence using python-docx."""
import json
from pathlib import Path
import statistics as st
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables'


def main():
    v1=json.loads((ROOT/'outputs/research/v1/research-results/report.json').read_text(encoding='utf-8'))
    v2=json.loads((ROOT/'outputs/research/v2/research-results/report-v2.json').read_text(encoding='utf-8'))
    v3path=ROOT/'outputs/research/v3/research-results/report-v3.json'
    v3=json.loads(v3path.read_text(encoding='utf-8')) if v3path.exists() else None
    doc=Document();md=[]
    section=doc.sections[0];section.page_width=Inches(8.5);section.page_height=Inches(11)
    section.top_margin=Inches(.75);section.bottom_margin=Inches(.75)
    section.left_margin=section.right_margin=Inches(.85)
    for style in ['Normal','Title','Heading 1','Heading 2','Caption']:
        s=doc.styles[style];s.font.name='Microsoft YaHei';s.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
        s.font.color.rgb=RGBColor(0,0,0)
    normal=doc.styles['Normal'];normal.font.size=Pt(10.5);normal.paragraph_format.line_spacing=1.3
    normal.paragraph_format.space_after=Pt(7)
    doc.styles['Title'].font.size=Pt(22)
    doc.styles['Heading 1'].font.size=Pt(15)
    doc.styles['Heading 2'].font.size=Pt(12)
    footer=section.footer.paragraphs[0];footer.alignment=2
    footer.add_run('曦源图像拼接软件 0.2.0  |  ')
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
    doc.core_properties.title='曦源图像拼接软件研究报告'
    doc.core_properties.author='曦源项目'
    def title(text):doc.add_heading(text,0);md.append('# '+text)
    def h(text):doc.add_heading(text,1);md.append('\n## '+text)
    def p(text):doc.add_paragraph(text);md.append(text)
    def table(headers,rows):
        t=doc.add_table(rows=1,cols=len(headers));t.style='Light Shading Accent 1'
        # Monochrome headings and white body; no decorative title rule.
        for c,text in zip(t.rows[0].cells,headers):c.text=str(text)
        repeat=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(repeat)
        for row in rows:
            for c,text in zip(t.add_row().cells,row):c.text=str(text)
        for row in t.rows:
            cannot=OxmlElement('w:cantSplit');row._tr.get_or_add_trPr().append(cannot)
            for c in row.cells:
                for para in c.paragraphs:
                    para.paragraph_format.space_after=Pt(5);para.paragraph_format.space_before=Pt(5)
                    for run in para.runs:run.font.size=Pt(9.5)
        md.extend(['| '+' | '.join(map(str,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+
                  ['| '+' | '.join(map(str,row))+' |' for row in rows])
    def page():doc.add_page_break()
    def figure(path,caption):
        doc.add_picture(str(path),width=Inches(6.7));doc.add_paragraph(caption,'Caption');md.append(caption)
    def mean_score(report,method,region):
        values=[r['quality'][method][region]['lpips'] for r in report['records'] if method in (r.get('quality') or {})]
        return st.mean(values) if values else None
    title('曦源图像拼接软件研究报告')
    p('用途：项目参与者交付与结题评审。记录日期：2026 年 9 月 10 日。依据原项目申请书推进；本报告记录已完成的软件、可复现实验和仍未得到支持的效果假设。')
    h('研究结论与交付范围')
    p('桌面软件已具备多图导入、自动配准、接缝画笔、传统融合对比和原尺寸导出。SD 1.5、ControlNet 与 LCM 已在免费 Kaggle T4 上真实运行，桌面任务到云端再回传的流程已验证。当前受控实验没有证明 AI 修复优于泊松融合，因此不将效果目标列为达成。')
    p('本次成果包括 Windows CPU 客户端、Python 源码与安装依赖、固定模型来源、GPU 运行脚本、实验原始记录和研究说明。客户端通过文件交接调用免费 GPU；它不是打包了模型的离线 AI 安装包。')
    figure(ROOT/'outputs/desktop-acceptance/returned.png','图 1  免费 GPU 返回的真实 AI 结果已在桌面打开')
    p('已交付软件功能与研究有效性是两项独立判断。下文将功能验证、质量评价和资源消耗分别报告，供负责人决定结题陈述和后续改进范围。')
    page();h('方法与实现')
    p('配准采用 SIFT 或 LoFTR 匹配，再使用 RANSAC 估计单应矩阵。图像映射到统一画布，累计有效像素区域。多图按用户指定的相邻重叠顺序增量拼接；该实现仍依赖单应模型，不是全局束调整或完整三维重建。')
    p('接缝采用最小代价路径，失败时退回重叠区中线。用户可通过画笔、橡皮和撤销调整修复区域。羽化和 OpenCV seamlessClone 泊松融合作为实际运行的传统基线。')
    p('扩散修复采用 SD 1.5 Inpainting，支持 Canny 和 Tile 两种 ControlNet，LCM LoRA 可缩短有效去噪步数。画布上的局部包围框经等比例缩放和补边进入 512 像素 Patch，生成后映射回原坐标。AI 输出按软 Mask 混合，Mask 外显式恢复底图，避免浮点取整引入额外像素变化。')
    p('实验性结构底图先根据输入重叠区域做稳健曝光估计，再尝试经双向一致性检查的 DIS 光流，并以窄接缝合成。该模块不接触参考全景。只有当重叠区域的像素一致性改善时才接受局部光流；这不等同于语义或几何正确性保证。默认关闭。')
    h('实验设计与评价口径')
    table(['实验','样本与比较','用途'],[
        ['v1','6 组参考控制对和 8 组实拍对；5 种 AI 配置','验证基本算法与初始消融'],
        ['v2','上述 14 组加 16 组局部形变派生对；4 种 AI 配置','检验结构底图及保守混合'],
        ['v3','30 组实拍对及 24 组预留扰动；3 种固定 AI 配置','扩展评估及 4K 多图压力测试']])
    p('受控样本来自砖墙、草地、碎石及含木桌的 coffee 照片。派生样本不是独立拍摄。v3 的预留扰动仍来自开发中使用过的原始照片，不能声称跨来源泛化；30 组实拍中有 8 组曾用于开发，其余按清单固定。真实照片可能存在跨仓库近似场景。')
    p('质量采用同一参考有效区域下的 MAE、SSIM 与空间 AlexNet LPIPS；LPIPS 越低越好。v1 评价接缝 ROI，v2/v3 同时评价全图和接缝。真实实拍没有完整真值，不计算虚假的参考图指标。运行成功只表示通过预设程序检查，不能代替人工几何检查。')
    page();h('开发实验质量结果')
    p('v1 共完成 70 次真实 AI 推理，无失败。6 组参考案例中，没有任何 AI 配置胜过羽化。其接缝 LPIPS 平均值如下；同一传统基线仅统计一次，不因多种 AI 配置重复加权。')
    baseline_rows=[r for r in v1['records'] if r.get('profile')=='plain' and r.get('quality')]
    rows=[]
    for method in ['Feather','Poisson','plain','canny','tile','canny_lcm','tile_lcm']:
        values=[r['quality'][method]['lpips_alex_spatial'] for r in (baseline_rows if method in ['Feather','Poisson'] else v1['records']) if method in (r.get('quality') or {})]
        rows.append([method,f'{st.mean(values):.5f}',len(values)])
    table(['方法','接缝 LPIPS 均值','参考案例数'],rows)
    p('v2 共 30 组案例，120 次真实 AI 推理全部成功；22 组有参考图。AI 混合比例固定 0.2，常规 Canny/Tile 为 20 步、强度 0.25，LCM 为 8 步、强度 0.5。')
    rows=[]
    for method in ['Feather','Poisson','Structural','raw_canny','refined_canny','refined_tile','refined_tile_lcm']:
        rows.append([method,f'{mean_score(v2,method,"full"):.5f}',f'{mean_score(v2,method,"seam"):.5f}'])
    table(['方法','全图 LPIPS','接缝 LPIPS'],rows)
    p('v2 的所有 AI 配置均没有在接缝 LPIPS 上胜过泊松；refined_canny 仅在 2/22 个案例胜过羽化，且 0/22 胜过自身结构底图。降低生成占比减少了修改，但没有建立 AI 本身带来质量收益的证据。不能将结构或曝光处理的个别改善归因于扩散模型。')
    page();h('固定参数扩展评估')
    if v3:
        success=sum(r.get('status')=='success' for r in v3['records'] if r.get('profile') in ['raw_canny','refined_canny','refined_tile_lcm'])
        count=sum(r.get('profile') in ['raw_canny','refined_canny','refined_tile_lcm'] for r in v3['records'])
        p(f'v3 运行状态为 {v3["status"]}。预设的 AI 评估记录共 {count} 项，成功 {success} 项。采用 v2 固定配置，未为测试案例逐一调参。每组实拍原始字节经 SHA256 检查，开发阶段长边上限 1200，扩展阶段长边上限 1600。')
        rows=[]
        for method in ['Feather','Poisson','Structural','raw_canny','refined_canny','refined_tile_lcm']:
            full=mean_score(v3,method,'full');seam=mean_score(v3,method,'seam')
            rows.append([method,f'{full:.5f}' if full is not None else '无有效结果',f'{seam:.5f}' if seam is not None else '无有效结果'])
        table(['方法','预留扰动全图 LPIPS','预留扰动接缝 LPIPS'],rows)
        for method in ['sift','loftr']:
            real=[r for r in v3['records'] if r.get('profile')=='structural_only' and 'perturbations' not in r.get('split','')]
            successes=sum(r.get('registration_comparison',{}).get(method,{}).get('status')=='success' for r in real)
            p(f'真实照片的 {method.upper()} 配准程序成功率为 {successes}/{len(real)}。该值基于匹配点、内点比例、画布和重叠阈值，不代表每幅图的视觉拼接均正确。')
        failures=[r for r in v3['records'] if r.get('status')!='success']
        if failures:
            p('失败记录保留在各案例 error.json 和总报告中。示例：'+ '; '.join(f'{r["id"]}：{r.get("error","配准未通过")[:100]}' for r in failures[:3]))
        else:p('本轮程序未记录计算失败。质量退化仍是失败案例的一种，不能由“零运行异常”推断零质量问题。')
        hr=v3.get('high_resolution',{})
        p('完整算法 4K 多图压力测试：'+json.dumps(hr,ensure_ascii=False)[:80] if hr.get('status')!='success' else
          f'完整算法 4K 多图压力测试成功，输出 {hr["shape"][1]}×{hr["shape"][0]}。这是程序生成的工程用例，只验证尺寸和运行资源，不能用作真实画质证据。')
    else:
        p('v3 已提交，结果尚未纳入本版。不得将待完成评估作为已经通过的验收证据。')
    h('真实照片人工评价')
    p('提供匿名左右对照材料及评分导出功能，比较接缝连续、纹理保持和整体偏好。当前尚无真人盲评结果，偏好率留空；本报告没有把自动指标换算为用户偏好，也没有伪造 70% 等目标值。评审时应按原始场景分组，并报告评分人数与无效票。')
    page();h('性能和软件验收')
    rows=[]
    for profile in ['raw_canny','refined_canny','refined_tile','refined_tile_lcm']:
        rs=[r for r in v2['records'] if r.get('profile')==profile and r['status']=='success']
        rows.append([profile,f'{st.median(r["metrics"]["inference_seconds"] for r in rs):.3f}',f'{max(r["metrics"]["peak_vram_mb"] for r in rs):.0f}'])
    table(['v2 配置','热推理中位数 秒','峰值分配 MiB'],rows)
    p('上述测试使用 Tesla T4 并设置 8 GiB PyTorch 分配上限。实际 T4 显存更多，缓存、驱动占用和其他进程另计；该结果不能替代原生 8GB 消费级显卡实测。LCM 的速度收益成立于本次配置，但其质量未优于传统基线。')
    p('桌面任务回传实测：首次模型加载 61.49 秒，推理 3.93 秒，AI 分支约 65.70 秒；配准坐标和手绘 Mask 保持一致，软 Mask 外逐像素相同。该时间不包含平台排队、Notebook 启动和网络传输。')
    p('Windows 本机三图工程用例输出 4096×1802，拼接 5.79 秒，包含保存和校验约 10.09 秒，进程峰值工作集约 796 MiB。免安装 EXE 已独立启动并显示 7 个查看页；wheel 已在项目目录外新虚拟环境安装并完成配准与 GUI 回放。项目交付相关自动化检查 24 项通过。')
    h('复现材料和局限')
    p('使用说明、依赖锁、模型来源和完整运行参数随交付提供。原始实验目录及每轮源码快照保存在 outputs/research；run.json 含图像哈希、模型版本、变换矩阵和计时。失败日志与基线结果一并保留。具体命令见 docs/delivery/复现与验收.md。')
    p('限制包括单应模型对大视差的适用范围、生成模型改动精细纹理、不同底图误差对全图评价的影响、来源数量少、实拍无参考全景和人工评价缺失。固定 Patch 保留导出尺寸，不代表全幅原生高清生成；多接缝共用包围框时有效细节可能减少。')
    p('本任务新增现金支出为 0 元，使用免费 GPU 额度。团队历史支出、人工、电费与既有设备未统计。原生 8GB 验证和真实用户偏好仍需补充；效果改进结论必须由后续证据决定，不能因软件功能完成而预先宣布。')
    h('参考来源')
    p('原项目申请书；LoFTR（CVPR 2021）；Latent Diffusion Models（CVPR 2022）；ControlNet（ICCV 2023）；Latent Consistency Models 与 LCM LoRA；Kornia、Diffusers、OpenCV 和 LPIPS 官方实现。模型精确 revision 与数据作者链接见《模型与第三方材料》及数据清单。')
    OUT.mkdir(exist_ok=True)
    doc.save(OUT/'曦源图像拼接研究报告.docx')
    (ROOT/'docs/delivery/研究报告.md').write_text('\n\n'.join(md)+'\n',encoding='utf-8')


if __name__=='__main__':main()
