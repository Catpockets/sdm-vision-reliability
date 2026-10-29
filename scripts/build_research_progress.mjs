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
const presentation2 = process.argv.includes('--presentation2');
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
if (presentation2) {
 const original = [...p.slides.items];
 original[0].shapes.items[2].text.replace('Clean-data findings and evaluation audit','Presentation 2: preliminary research findings');
 text(original[2],'Compute: local MPS with CPU/CUDA support. No cloud deployment is claimed.',70,615,1120,35,21,GRAY);
 const problem=slide('A review policy for confident vision errors','');
 text(problem,'Target user',70,185,500,55,30,BLUE,true);
 text(problem,'Vision practitioners deciding which model outputs can be accepted and which need human review.',70,270,510,170,32,GRAY);
 text(problem,'Why coverage matters',680,185,500,55,30,BLUE,true);
 text(problem,'A rule that rejects everything has no practical utility. A rule that accepts everything inherits the model’s errors.',680,270,500,210,32,GRAY);
 text(problem,'CIFAR-100 example: 85.09% accepted, 14.91% deferred, 95.06% accuracy among accepted predictions.',70,545,1120,100,28,BLUE);
 const paper=slide('A reproducible benchmark and paper','');
 text(paper,'Research deliverable',70,185,530,60,30,BLUE,true);
 text(paper,'Matched-checkpoint comparisons\nRunnable evaluation notebooks\nCoverage and error-analysis tables\nA paper reporting positive and negative results',70,275,540,255,30,GRAY);
 text(paper,'Practitioner questions',680,185,500,60,30,BLUE,true);
 text(paper,'What coverage is attainable at a target accepted accuracy?\n\nWhen does SDM add value beyond calibrated softmax?',680,275,510,255,30,GRAY);
 text(paper,'Practitioner and expert validation remains a task unless the team confirms completed feedback.',70,600,1120,48,23,GRAY);
 const method=slide('Similarity, distance and magnitude in SDM','');
 ['Similarity','Distance','Magnitude'].forEach((v,i)=>text(method,v,70+390*i,190,335,65,31,BLUE,true));
 ['Agreement with nearby training examples in representation space','Proximity to the training distribution','Decision-boundary information from the model’s logits'].forEach((v,i)=>text(method,v,70+390*i,280,335,180,30,GRAY));
 text(method,'Calibration constructs a selective region. A region can be empty when its empirical conditions cannot be met.',70,535,1110,100,30,BLUE);
 const plan=slide('Remaining work and proposed owners','');
 table(plan,[['Workstream','Proposed owner','Next deliverable'],['SDM and protocol','Yiwen','Freeze choices and explain region diagnostics'],['Dataset comparisons','Randell','Reconcile run artifacts and coverage results'],['Baselines and paper','Karim','Integrate comparisons, figures and limitations']],190,320,[290,220,630]);
 text(plan,'Priority: matched CIFAR corruption/OOD evaluation, then unfinished dataset checks. Full runs remain paused.',70,550,1110,95,27,GRAY);
 const mission=slide('Our mission','');
 text(mission,'Give vision practitioners reproducible evidence for when a classifier should answer, abstain, or defer to human review.',90,230,1080,240,46,BLUE);
 text(mission,'Reliable accepted predictions at useful coverage',90,555,1080,65,30,GRAY);
 const selected=[original[0],problem,paper,method,original[2],original[4],original[3],original[5],original[6],original[8],original[10],plan,mission];
 const allSlides=[...p.slides.items]; p.slides.keep(selected.map(s=>allSlides.indexOf(s)));
 const records=(await p.inspect({kind:'slide',maxChars:15000})).ndjson.trim().split('\n').map(line=>JSON.parse(line));
 const canonical=new Map(p.slides.items.map((s,i)=>[s.id,records.find(r=>r.slideIndex===i).id]));
 p.slides.reorder(selected.map(s=>canonical.get(s.id)));
 const talk=[
 ['Karim','0:10','Our project asks when a vision classifier should answer and when it should abstain. We will explain the research deliverable, the experiment design, our completed clean-data results and the evidence still needed before broader reliability claims.'],
 ['Karim','1:50','The target audience is a vision practitioner choosing an acceptance and review policy. Classification accuracy averages across all examples. It does not tell us whether a particular confidence rule can identify a useful subset of reliable predictions. Accepting every example preserves the original error rate. Rejecting every example avoids accepted errors but also provides no classification utility. We therefore report accepted accuracy and coverage together. In our completed CIFAR-100 ConvNeXt test, the standard-track temperature rule accepts 8,509 of 10,000 images, or 85.09%, at 95.06% accepted accuracy. The remaining 1,491 images would form a potential review pool in a hypothetical workflow. We have not measured a real inspection process, human accuracy, operational savings or deployment safety. Our benchmark is a way to compare policies before making those claims.'],
 ['Karim','2:45','Our deliverable is a research paper supported by reproducible experiments. It serves practitioners by answering two questions: what coverage is attainable at a chosen empirical reliability target, and whether SDM adds value beyond cheap raw-softmax and temperature-scaled baselines. We preserve the same checkpoint, label space and evaluation images within a comparison, record calibration support, and retain negative results. The supporting notebooks export configurations, identities, per-image decisions and aggregate metrics. The paper connects those outputs to accepted accuracy, coverage, probability calibration, error ranking, confusions and limitations. The project does not claim a production inspection product or a general safety guarantee. We considered several distinct confidence measurements because calibration quality, ranking quality and admission feasibility can disagree. A useful final contribution can be a carefully evidenced limitation of SDM transfer, not only a positive result. The full corrupted/OOD comparison remains unfinished, so today’s conclusion is deliberately restricted to completed clean tasks. The repository does not establish that the planned expert meeting occurred. Do not claim an interview, quote, or user validation unless the team confirms it. Questions for expert feedback include why a qualifying region can fail despite useful error ranking, and how to compare global empirical thresholds with SDM’s class-conditional construction.'],
 ['Yiwen','1:15','SDM uses representation-based signals rather than softmax confidence alone. Similarity reflects correctly predicted depth matches to training examples; distance reflects proximity to the training distribution; magnitude uses decision-boundary information. Calibration combines these signals into a selective admission region. Our implementation tracks must remain distinct: ResNet uses a learned adaptor through reexpress-sdm, whereas the completed ConvNeXt comparisons use the identity-adaptor formulation. Those variants are not a controlled architecture ablation. A region may be empty. That means zero coverage and undefined accepted accuracy, not a successful 95% guarantee. Source for the method terminology: docs/allen-schmaltz-sdm.md and the pinned upstream implementation.'],
 ['Yiwen','1:00','The diagram shows the EMNIST population. Training has 106,925 examples, validation 1,175, enlarged calibration 4,700 and official test 18,800. Standard calibration contains 1,175 examples nested inside the enlarged set. Validation selects the ConvNeXt checkpoint. Calibration fits the temperatures and admission rules. Each comparison keeps the checkpoint, classes and evaluation population fixed. The ConvNeXt models use ImageNet-pretrained ConvNeXt-Tiny, seed 42, 20 epochs and validation-best checkpoint selection. Compute uses local Apple MPS and saved artifacts; the pipeline supports CPU and CUDA. Cloud CUDA is an alternative for later runs, not an existing deployed service. Local execution makes current cached artifacts available but limits throughput. No new compute run is launched here. The from-scratch ResNet uses its own recipe and reports validation, so we do not pool its results with ConvNeXt official-test results. Current analysis follows historical test exploration and is not a blind preregistration.'],
 ['Karim','0:50','Positive scalar temperature scaling changes probabilities without changing the predicted class. It improves ECE on all five completed ConvNeXt tasks shown here, using 25 calibration images per class. ECE is shown as a percentage with 15 bins. CIFAR-100 changes from 5.96% to 3.36%; EMNIST from 6.29% to 2.94%. Lower clean ECE is useful, but it does not imply that every other probability metric improves or that behavior remains reliable under distribution shift. We retain raw softmax because it is the simplest baseline and use calibration-only temperature fitting rather than fitting against test labels.'],
 ['Randell','1:10','These are five distinct official-test tasks and checkpoints, not one pooled model comparison. CIFAR-100 has 88.45% classification accuracy; Fashion-MNIST 95.02%; EMNIST 90.38%. Neither calibration budget produces a qualifying SDM region on those three tasks. MNIST admits 60.69% and 49.81% of its test set with no observed admitted errors. KMNIST admits 33.14% and 39.92%, with zero and two admitted errors respectively. More calibration support does not monotonically increase coverage. Full-coverage classification accuracy alone also does not determine whether a region forms: Fashion-MNIST exceeds 95% accuracy but has no region. These observations are descriptive; they do not isolate class count, resolution, or another mechanism. We should distinguish these verified ConvNeXt results from older teammate ResNet/Imagenette experiments whose numerical reconciliation remains pending.'],
 ['Randell','0:50','EMNIST separates the three measurement questions. Its enlarged SDM track has AURC 0.01912, lower than raw softmax’s 0.03964, so its retrospective error ranking is better under that metric. Yet its native exported ECE is 51.30%, compared with 6.29% for raw softmax, and neither SDM track yields a qualifying admission region. A ranking improvement cannot stand in for a usable frozen policy. The SDM probability-matrix metrics are reconciled native exports; ranking and admissions were recomputed from saved per-image scores. One seed and one nested calibration comparison do not establish a general architecture or class-count effect.'],
 ['Karim','0:45','Here the thresholds come from calibration only, with at least 100 admitted calibration examples and whole tied-confidence groups. The displayed values apply to saved official-test predictions. EMNIST’s 25/class temperature rule admits 84.63% at 96.17% accepted accuracy, while its 100/class rule admits 89.81% at 94.56%. The latter misses the 95% target. This is evidence of the behavior of these empirical policies, not evidence that more calibration data causes lower reliability. Exact counts and Wilson intervals are in the paper. A global empirical accuracy threshold also differs from SDM’s class-conditional region construction.'],
 ['Karim','0:40','These are the three highest standard-track temperature-confidence errors in CIFAR-100, selected deterministically with test ID as the tie-breaker. They illustrate remaining confident mistakes even after aggregate calibration improves. The otter is predicted as seal at approximately 0.992 confidence, the baby as girl at 0.986, and the kangaroo as rabbit at 0.983. We checked the IDs and labels against the cached dataset. These selected cases are not an estimate of overall error frequency or proof of label noise.'],
 ['Karim','0:45','The audit keeps the comparison interpretable: equal confidence scores enter together; calibration-only matching does not use validation coverage; saved SDM fits must match the calibration population. Historical source repairs do not by themselves validate old numerical results. The main scientific limitations remain one seed, different backbone recipes and reporting populations, historical test exploration, and unfinished matched corruption/OOD evaluation. We can describe the clean findings today without claiming that they establish deployment safety or robustness.'],
 ['Karim','0:30','This is a proposed workstream plan based on our current contributions and speaking topics, not a claim that new assignments were already agreed. Yiwen leads the SDM/protocol explanation; Randell the dataset coverage comparisons; Karim the baselines, evidence reporting and paper integration. The immediate deliverables are reconciled run artifacts and frozen evaluation choices. The remaining experiment priority is matched CIFAR-100-C and SVHN-OOD evaluation, followed by unfinished in-distribution checks. Full runs remain paused and this slide does not restart them. Confirm ownership and execution timing as a team.'],
 ['Karim','0:30','Our mission is to give vision practitioners reproducible evidence for when a classifier should answer, abstain or defer to review. Useful coverage and reliable accepted predictions must occur together. Our completed clean experiments show mixed SDM transfer behavior, including empty regions, and a clear role for calibrated baselines. The final paper will report those findings and the remaining evaluation honestly. Invite questions without adding a thank-you slide.']
 ];
 selected.forEach((s,i)=>s.speakerNotes.text=`Suggested speaker: ${talk[i][0]}. Rehearsal target: ${talk[i][1]}.\n\n${talk[i][2]}\n\nSources: docs/assets/clean-analysis-2026-10-29/evidence.json (SHA256 ${sourceHash}); docs/research-writeup.md; docs/experiment-design.md. Presentation 2 guidance supplied by the team, pages 1–5. Speaking assignments and timings are suggestions based on the chat, not confirmed team decisions.`);
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
 explicitTotalSlideCount:presentation2?13:12,requiredNativeTableOwnerSlides:presentation2?[7,9,12]:[4,7,8],requiredNativeChartOwnerSlides:[presentation2?6:5],
 materializeLiteralChartWorkbooks:true,
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...((presentation2?[7,9,12]:[4,7,8]).flatMap(n=>['--require-native-table-slide',String(n)]))],
 fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,
 receiptPath:path.join(build,'validation.json')});
console.log('Validated deck:',final);
