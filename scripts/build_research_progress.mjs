// Local PowerPoint authoring with @oai/artifact-tool. No model execution.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
const {Presentation, PresentationFile} = await import(process.env.ARTIFACT_TOOL_MODULE ?? '@oai/artifact-tool');
if (!process.env.RUNTIME_NODE_MODULES && process.env.ARTIFACT_TOOL_MODULE?.startsWith('/')) process.env.RUNTIME_NODE_MODULES = path.resolve(path.dirname(process.env.ARTIFACT_TOOL_MODULE), '../../..');
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SKILL = process.env.PRESENTATIONS_SKILL_DIR;
if (!SKILL) throw Error('Set PRESENTATIONS_SKILL_DIR to the installed Presentations skill');
const {finalizePresentation, applyPresentationChartFont} = await import(pathToFileURL(path.join(SKILL, 'container_tools/artifact_tool_utils.mjs')));
const build = path.resolve(process.argv[2] ?? path.join(ROOT, '.presentation-build'));
const final = path.resolve(process.argv[3] ?? path.join(ROOT, 'presentations/sdm-research-progress.pptx'));
await fs.mkdir(build, {recursive:true}); await fs.mkdir(path.dirname(final), {recursive:true});
const evidencePath = path.join(ROOT, 'docs/assets/clean-analysis-2026-10-29/evidence.json');
const evidenceBytes = await fs.readFile(evidencePath);
const evidence = JSON.parse(evidenceBytes);
const sourceHash = createHash('sha256').update(evidenceBytes).digest('hex');
const FONT = 'Arial', BLUE='#002676', GOLD='#FDB515', GRAY='#595959';
const p = Presentation.create({slideSize:{width:1280,height:720}});
const pct = x => x == null ? 'Undefined' : (x*100).toFixed(2)+'%';
const policy = (g,method,track) => g.policies.find(r=>r.method===method && r.track===track);
function text(s, str, x,y,w,h,size=28,color=BLUE,bold=false){
  const box=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  box.text=[[{run:str,style:{fontSize:size+'px',typeface:FONT,color,bold}}]];
  box.text.style={fontSize:size,alignment:'left',verticalAlignment:'middle',wrap:true,insets:0};
  return box;
}
function slide(title,notes){
  const s=p.slides.add(); s.background.fill=GOLD;
  s.shapes.add({geometry:'rect',position:{left:24,top:24,width:1232,height:672},fill:'#FFFFFF',line:{fill:'none',width:0}});
  text(s,title,70,60,1140,100,40,BLUE,true);
  s.speakerNotes.text=notes+'\n\nSource: docs/assets/clean-analysis-2026-10-29/evidence.json; SHA256 '+sourceHash+'. Campaign source '+evidence.source_revision+'.\nAdditional sources: docs/research-writeup.md; docs/experiment-design.md. Visual reference: team SDM deck, https://docs.google.com/presentation/d/1aMHqcsG_W8iX00KrW3dlTW-E8DPxAxeZ/edit (September 29 version).';
  return s;
}
function table(s,values,y=180,height=390,widths){
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:70,top:y,width:1140,height,
    ...(widths?{columnWidths:widths}:{}),values});
  t.borders.assign({style:'solid',fill:'#DDDDDD',width:.6});
  values.forEach((row,r)=>row.forEach((v,c)=>{
    const cell=t.getCell(r,c);cell.fill=r===0?BLUE:'#FFFFFF';
    cell.text.set([[{run:String(v),style:{typeface:FONT,fontSize:'23px',bold:r===0,color:r===0?'#FFFFFF':'#222222'}}]]);
    cell.text.style={fontSize:23,insets:12,verticalAlignment:'middle'};
  }));return t;
}
async function picture(s,relative,x,y,w,h){
  const b=await fs.readFile(path.join(ROOT,relative));
  s.images.add({blob:b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength),contentType:'image/png',alt:relative,fit:'contain',position:{left:x,top:y,width:w,height:h}});
}
{
 const s=slide('SDM Vision Reliability','Opening: present this as a progress report, not a final robustness claim. The work asks when an image classifier should abstain. Distinguish the original proposal from the measured clean results. The experiment campaign is paused; all new analysis reads existing artifacts. Suggested time: 45 seconds.');
 text(s,'Clean-data findings and evaluation audit',70,230,1080,90,44,BLUE);
 text(s,'Randell Smith · Yiwen Hou · Karim Mattar\nUC Berkeley MIDS Capstone · October 29, 2026',70,365,1080,100,28,GRAY);
 text(s,'Six completed classifiers · one seed per task · shift evaluation unfinished',70,555,1100,65,26,BLUE);
}
{
 const s=slide('Prediction accuracy does not measure selective reliability','Explain accuracy as correctness across all examples, ranking as whether confidence orders errors below correct predictions, and admission as the behavior of a fixed abstention policy. An empty region has zero coverage and undefined accepted accuracy. These distinctions explain why a ranking improvement can coexist with an unusable deployed rule. Suggested time: 70 seconds.');
 ['Classification','Error ranking','Frozen admissions'].forEach((t,i)=>text(s,t,70+i*390,200,340,65,30,BLUE,true));
 ['How often is the predicted class right?','Does confidence put errors below correct predictions?','Which inputs does a calibration-selected rule accept?'].forEach((t,i)=>text(s,t,70+i*390,290,330,155,29,GRAY));
 text(s,'Report accepted accuracy together with coverage. Zero admissions do not establish reliability.',70,530,1100,100,30,BLUE,true);
}
{
 const s=slide('Compare methods within a fixed classifier and population','Methods share one checkpoint and evaluation population within each block. ResNet is from scratch and reports validation; ConvNeXt is pretrained and reports official test. Training support, transforms, and selection rules differ. Do not treat the architecture comparison as a controlled uncertainty-method ablation. The numbers shown here are the EMNIST protocol. Suggested time: 75 seconds.');
 text(s,'EMNIST Balanced protocol',70,165,1100,35,23,GRAY);
 const stages=[['Training','106,925 images'],['Validation','1,175 images'],['Calibration','1,175 / 4,700'],['Official test','18,800 images']];
 stages.forEach(([a,b],i)=>{text(s,a,70+i*290,210,255,70,30,BLUE,true);text(s,b,70+i*290,300,255,70,26,GRAY);if(i<3)text(s,'→',315+i*290,285,35,55,32,BLUE);});
 text(s,'Validation selects the checkpoint; calibration selects the confidence rule.\nThe standard calibration set is nested inside the enlarged set.',70,445,1100,120,29,GRAY);
}
{
 const s=slide('Five completed ConvNeXt tasks show mixed SDM coverage','These five official-test populations are separate tasks, each with its own model. All use one backbone seed. Also completed: a from-scratch ResNet CIFAR-100 classifier and learned SDM fits; those use a different clean reporting population. Do not rank models across the rows. No-region outcomes are retained. Suggested time: 70 seconds.');
 table(s,[['Dataset','Classes','Test images','Accuracy','SDM coverage\n25 / 100 per class'],...evidence.groups.map(g=>[g.title,String(g.classes),g.examples.toLocaleString('en-US'),pct(g.accuracy),pct(policy(g,'sdm_identity','standard_25').coverage)+' / '+pct(policy(g,'sdm_identity','enlarged_100').coverage)])],180,390,[280,130,180,170,380]);
 text(s,'ResNet is reported separately on validation; its results are not pooled with these test populations.',70,600,1120,50,23,GRAY);
}
{
 const s=slide('Temperature scaling lowers clean calibration error','Show ECE in percentage points. Raw softmax and temperature-25 predictions share the same labels and checkpoint; temperature changes confidence rather than top-1 decisions. These are descriptive single-seed results. Lower ECE does not establish robustness or imply every metric improves. Suggested time: 70 seconds.');
 const chart=s.charts.add('bar',{position:{left:70,top:175,width:1130,height:410},
   categories:evidence.groups.map(g=>g.title.replace(' Balanced','')),
   series:[{name:'Raw softmax',values:evidence.groups.map(g=>Number((100*g.native_metrics.find(r=>r.method==='softmax').ece_15).toFixed(2))),fill:BLUE},
           {name:'Temperature, 25/class',values:evidence.groups.map(g=>Number((100*g.native_metrics.find(r=>r.method==='temperature_standard_25').ece_15).toFixed(2))),fill:GOLD}],
   hasLegend:true,dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.00',textStyle:{fontSize:20}},xAxis:{textStyle:{fontSize:20}},yAxis:{numberFormatCode:'0.0',textStyle:{fontSize:18},majorGridlines:{fill:'#EEEEEE',width:1}}});
 applyPresentationChartFont(chart,{fontFamily:FONT});
 text(s,'15-bin ECE (%) · same checkpoint and class predictions within each task',70,610,1120,40,23,GRAY);
}
{
 const s=slide('EMNIST ranking improves while both SDM regions stay empty','EMNIST has 47 classes and 18,800 test examples. SDM-100 has lower AURC than raw softmax, but its probability calibration is worse and no qualifying region forms. ECE values on this figure use rates, unlike the previous slide percentages. SDM probability metrics are reconciled native exports; ranking and admissions were independently recomputed from per-image saved scores. Suggested time: 80 seconds.');
 await picture(s,'docs/assets/emnist-clean-2026-10-29.png',60,160,1160,455);
}
{
 const s=slide('A 95% calibration target can miss on held-out data','The table reports calibration-selected global temperature rules, not retrospective threshold tuning on test labels. The minimum admitted calibration count is 100. Larger calibration support does not guarantee higher test accepted accuracy under this empirical selection procedure. EMNIST-100 misses the target while admitting more test images. The comparison is retrospective and single-seed; it is not evidence that more calibration data causes worse performance. Suggested time: 80 seconds.');
 table(s,[['Dataset','Coverage\n25/class','Accuracy\n25/class','Coverage\n100/class','Accuracy\n100/class'],...evidence.groups.map(g=>{const a=policy(g,'temperature','standard_25'),b=policy(g,'temperature','enlarged_100');return [g.title,pct(a.coverage),pct(a.accepted_accuracy),pct(b.coverage),pct(b.accepted_accuracy)];})],180,390,[290,210,210,215,215]);
 text(s,'Temperature-scaled softmax rules; exact counts and Wilson intervals are in the write-up.',70,603,1120,48,23,GRAY);
}
{
 const g=evidence.groups.find(g=>g.dataset==='emnist_balanced');
 const s=slide('EMNIST errors concentrate in similar-looking labels','Directed confusions count true class to predicted class separately; reciprocal errors are separate entries. Select pairs by largest error count, then class IDs for ties. This selection is descriptive after test exploration. Visual ambiguity is a hypothesis; it is not proof of incorrect annotation or the cause of SDM region failure. Suggested time: 60 seconds.');
 table(s,[['True class','Predicted class','Test errors'],...g.top_confusions.slice(0,5).map(r=>[r.true_class,r.predicted_class,String(r.count)])],190,350,[380,380,380]);
 text(s,'Largest directed error counts; these observations do not establish annotation errors or a causal mechanism.',70,585,1120,65,25,GRAY);
}
{
 const s=slide('High-confidence CIFAR errors remain after temperature scaling','Examples are the three highest temperature-standard confidence errors, with dataset ID breaking ties. They are deliberately selected errors and do not estimate the frequency of all errors. The image labels and predictions are checked against the saved official-test arrays and local dataset cache. Do not treat softmax confidence as an individual error guarantee. Suggested time: 60 seconds.');
 await picture(s,'docs/assets/clean-analysis-2026-10-29/cifar100-errors.png',65,175,1150,425);
}
{
 const s=slide('EMNIST confusions remain visible at high confidence','The same deterministic example selection is applied to EMNIST. The images are displayed upright using the same transpose as the experiment. The examples are qualitative illustrations, not independent evidence of label noise or a class-count mechanism. Suggested time: 60 seconds.');
 await picture(s,'docs/assets/clean-analysis-2026-10-29/emnist_balanced-errors.png',65,175,1150,425);
}
{
 const s=slide('The audit changed how the evidence can be used','The tied-score threshold defect was reproduced in older ResNet notebooks 04–10. Coverage matching in 07/09/10 and class sweep had depended on validation coverage. Those code paths are repaired in PR 33, but source repair alone cannot revalidate the historical results. We retain negative findings and reject stale artifacts. Suggested time: 70 seconds.');
 text(s,'Corrected',70,190,500,60,30,BLUE,true);
 text(s,'Whole-score-group thresholds and risk curves\nCalibration-only matched coverage\nCached calibration identity checks\nEmpty regions and small galleries',70,280,540,270,29,GRAY);
 text(s,'Still limited',680,190,500,60,30,BLUE,true);
 text(s,'One seed per completed task\nDifferent ResNet / ConvNeXt populations\nHistorical test exploration\nNo completed matched shift benchmark',680,280,500,270,29,GRAY);
}
{
 const s=slide('Clean-data conclusions are ready; shift claims remain open','Close with the research question. Temperature scaling improves clean probability calibration on the completed tasks. SDM ranking and admission behavior must be reported separately, including no-region outcomes. Heavy experiments remain paused. Needed for the older teammate comparisons: exact checkpoint, split membership/class map, SDM artifact, and saved calibration/evaluation logits or scores. No expert meeting, new seed run, corruption result or OOD robustness result is implied. Suggested time: 60 seconds; then questions.');
 text(s,'Supported by saved results',70,190,1120,60,30,BLUE,true);
 text(s,'Temperature scaling improves clean calibration.\nSDM has mixed ranking results and task-dependent admission coverage.',70,270,1090,120,31,GRAY);
 text(s,'Before broader claims',70,440,1120,60,30,BLUE,true);
 text(s,'Reconcile teammate artifacts; then complete the paused corruption/OOD and remaining dataset runs.',70,520,1090,100,31,GRAY);
}
for (let i=0;i<p.slides.items.length;i++) {
 const blob=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(build,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const candidate=path.join(build,'candidate.pptx');await (await PresentationFile.exportPptx(p)).save(candidate);
await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:final,
 pythonExecutable:process.env.RUNTIME_PYTHON,
 integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
 explicitTotalSlideCount:12,requiredNativeTableOwnerSlides:[4,7,8],requiredNativeChartOwnerSlides:[5],
 materializeLiteralChartWorkbooks:true,
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...([4,7,8].flatMap(n=>['--require-native-table-slide',String(n)]))],
 fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,
 receiptPath:path.join(build,'validation.json')});
console.log('Validated deck:',final);
