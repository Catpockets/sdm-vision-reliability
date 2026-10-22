"""Shift evaluation rejects artifact drift and preserves honest abstention/OOD metrics."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import hashlib
import io
import tarfile
from unittest.mock import patch
from types import SimpleNamespace
import unittest

import numpy as np
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'notebooks'), str(ROOT/'scripts')]
from shift_metrics import admission_mask, artifact_identity, metric_functions, threshold_for_target, verify_identity
from evaluate_matched_shift import CorruptionSlice, summarize, validate_splits, verify_replay
from collect_experiment_results import collect, comparison_group
import run_shift_campaign


class ShiftEvaluationTest(unittest.TestCase):
    def test_legacy_notebooks_use_same_attainable_thresholds(self):
        for path in sorted((ROOT/'notebooks').glob('1[123]_cifar100*.ipynb')):
            namespace = {'calibrated_threshold': threshold_for_target, 'MIN_ADMITTED': 100}
            n=json.loads(path.read_text())
            for cell in n['cells']:
                if cell['cell_type'] == 'code':
                    source=''.join(cell['source']); compile(source,cell['id'],'exec')
                    nodes=[node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name=='threshold_for_target']
                    if nodes: exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
            correct=np.r_[np.ones(180,dtype=bool),np.zeros(20,dtype=bool)]
            self.assertIsNone(namespace['threshold_for_target'](np.full(200,.9), correct,.95))
            self.assertEqual(namespace['threshold_for_target'](np.full(200,.9),correct,.85),.9)
            self.assertIsNone(namespace['threshold_for_target'](np.ones(99),np.ones(99,dtype=bool),.95))

    def test_checkpoint_and_export_drift_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            names=['best_checkpoint.pt','calibration_outputs_full.npz','test_outputs_full.npz']
            for name in names:(root/name).write_bytes(b'original')
            identity=artifact_identity(root);verify_identity(root,identity)
            for name in names:
                (root/name).write_bytes(b'other checkpoint or outputs')
                with self.assertRaisesRegex(ValueError,'identity changed'):verify_identity(root,identity)
                (root/name).write_bytes(b'original')
            with self.assertRaises(ValueError):verify_identity(root,None)

    def test_split_overlap_and_mismatched_replayed_logits_rejected(self):
        valid={'train':np.array([0,1]),'validation':np.array([2]),'standard':np.array([3]),'enlarged':np.array([3,4])}
        validate_splits(valid)
        bad=dict(valid,validation=np.array([1]))
        with self.assertRaisesRegex(ValueError,'overlap'):validate_splits(bad)
        with self.assertRaisesRegex(ValueError,'nested'):validate_splits(dict(valid,standard=np.array([8])))
        expected=np.array([[1.,2.]])
        verify_replay(expected,expected,[1],[1])
        with self.assertRaisesRegex(ValueError,'does not reproduce'):verify_replay(expected+1,expected,[1],[1])
        with self.assertRaises(ValueError):verify_replay(expected,expected,[0],[1])

    def test_no_region_and_zero_threshold_are_distinct(self):
        np.testing.assert_array_equal(admission_mask([0,.5,1],None),[False]*3)
        np.testing.assert_array_equal(admission_mask([0,.5,1],0),[True]*3)

    def test_ood_never_becomes_cifar_accuracy_and_tied_auc_is_half(self):
        ns=metric_functions()
        experiment=SimpleNamespace(eval_split='validation',tracks={'standard_25':{'calibration_examples':2500}},ns=ns,
                                   clean_confidence={'standard_25__softmax':np.full(4,.5)})
        methods={('standard_25','softmax'):(np.full((4,2),.5),np.zeros(4,dtype=int),np.zeros(4,dtype=bool),100)}
        rows,individual=summarize(experiment,methods,None,'svhn_ood',{})
        row=rows[0]
        self.assertEqual(row['id_vs_ood_auroc'],.5)
        self.assertEqual(row['ood_false_admission_rate'],0)
        self.assertTrue(np.isnan(row['accepted_accuracy']))
        self.assertNotIn('full_coverage_accuracy',row)
        self.assertNotIn('nll',row)
        self.assertEqual(row['eval_split'],'svhn_test')
        self.assertEqual(row['id_reference_split'],'validation')

    def test_corruption_severity_offsets_and_input_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            images=np.lib.format.open_memmap(root/'gaussian_noise.npy',mode='w+',dtype=np.uint8,shape=(50000,32,32,3))
            for s in range(5):images[s*10000]=s+1
            images.flush();del images
            np.save(root/'labels.npy',np.tile(np.arange(10000)%100,5))
            for s in range(1,6):
                dataset=CorruptionSlice(root,'gaussian_noise',s,transforms.ToTensor())
                image,label=dataset[0]
                self.assertAlmostEqual(float(image[0,0,0]),s/255)
                self.assertEqual(label,0)
            with self.assertRaises(ValueError):CorruptionSlice(root,'missing',1,None)
            with self.assertRaises(ValueError):CorruptionSlice(root,'gaussian_noise',0,None)

    def test_comparison_groups_preserve_checkpoint_class_and_population(self):
        row=dict(dataset='cifar100',condition='clean',classes=100,eval_split='validation',
                 checkpoint_sha256='one',seed=42,track='standard_25',metric_definition='grouped-v1')
        original=comparison_group(row)
        self.assertEqual(original,comparison_group(dict(row,method='another method')))
        for key,value in [('checkpoint_sha256','two'),('classes',20),('eval_split','official_test'),('track','enlarged_100')]:
            self.assertNotEqual(original,comparison_group(dict(row,**{key:value})))

    def test_pending_and_smoke_results_are_not_reported_as_benchmarks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'status.json').write_text(json.dumps({'source_revision':'fixture','jobs':[{'id':'cifar100','status':'queued','log':'pending'}]}))
            smoke=root/'shift';smoke.mkdir()
            (smoke/'experiment_metadata.json').write_text(json.dumps({'status':'pipeline_check','mode':'smoke'}))
            rows,inventory=collect(root,[smoke])
            self.assertTrue(rows.empty)
            self.assertEqual(list(inventory.status),['queued','excluded_pipeline_check'])

    def test_archive_integrity_is_checked_before_safe_extraction(self):
        from evaluate_matched_shift import CORRUPTIONS
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            archive=root/'CIFAR-100-C.tar'
            with tarfile.open(archive,'w') as output:
                for name in [*(c+'.npy' for c in CORRUPTIONS),'labels.npy']:
                    item=tarfile.TarInfo('CIFAR-100-C/'+name)
                    payload=b'preparation fixture';item.size=len(payload)
                    output.addfile(item,io.BytesIO(payload))
            expected=hashlib.md5(archive.read_bytes()).hexdigest()
            with patch.object(run_shift_campaign,'SIZE',archive.stat().st_size), patch.object(run_shift_campaign,'MD5',expected):
                run_shift_campaign.prepare(root)
                self.assertEqual(json.loads((root/'CIFAR-100-C/SOURCE.json').read_text())['archive_md5'],expected)
                with archive.open('ab') as output:output.write(b'corruption')
                with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                    run_shift_campaign.prepare(root)

    def test_legacy_ood_and_corruption_summaries_allow_no_admissions(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import contextlib
        names={'12': ['5cd65bf1','80660daf'], '13':['f6a7b8c9','a7b8c9d0']}
        with tempfile.TemporaryDirectory() as directory, patch.object(plt,'show',side_effect=lambda:plt.close('all')):
            for number,cell_ids in names.items():
                path=next((ROOT/'notebooks').glob(number+'_cifar100*.ipynb'))
                notebook=json.loads(path.read_text())
                digits=type('Digits',(),{'labels':np.arange(5),'__len__':lambda self:5})()
                namespace=dict(np=np,json=json,plt=plt,RUN_DIR=Path(directory),TEMPERATURE=1.,
                               THRESHOLDS={'softmax':None,'temp-scaled':None},
                               softmax_with_temperature=metric_functions()['temperature_probabilities'],
                               metric_functions=metric_functions,svhn=digits,svhn_logits=np.zeros((5,100)),
                               test_raw=np.full((4,100),.01),test_scaled=np.full((4,100),.01),
                               cifar100_names=[str(i) for i in range(100)],CORRUPTIONS=('gaussian_noise',),
                               SEVERITIES=(1,),SCOPE_NAME='fixture',SCOPE={'name':'fixture'},DATA_SOURCE={'source':'fixture'},
                               CLEAN={'accuracy':0.,'softmax':0.,'temp-scaled':0.},
                               RESULTS={('gaussian_noise',1):(np.zeros((5,100)),np.arange(5))})
                with contextlib.redirect_stdout(io.StringIO()):
                    for cell in notebook['cells']:
                        if cell['id'] in cell_ids:
                            exec(compile(''.join(cell['source']),cell['id'],'exec'),namespace)
                if number=='12':
                    self.assertEqual(namespace['metrics']['metrics']['softmax']['admission_rate_at_frozen_threshold'],0)
                    self.assertEqual(namespace['metrics']['metrics']['softmax']['id_vs_ood_auroc'],.5)
                else:
                    self.assertEqual(namespace['METRICS'][('gaussian_noise',1)]['softmax']['admission_rate'],0)
                    self.assertTrue(np.isnan(namespace['METRICS'][('gaussian_noise',1)]['softmax']['accepted_accuracy']))


if __name__ == '__main__':unittest.main()
