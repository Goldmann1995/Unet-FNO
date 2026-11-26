import unittest
import numpy as np
import torch
from controlled_training import LowResolutionUNet,split_ids,channel_stats,temporal_indices,build_model,training_step,fit_normalization

class ControlledTests(unittest.TestCase):
    def test_saved_split_disjoint_complete_and_repeatable(self):
        ids=[f'sample_{i:03d}' for i in range(500)]
        tr,va=split_ids(ids)
        self.assertEqual((len(tr),len(va)),(400,100))
        self.assertFalse(set(tr)&set(va));self.assertEqual(set(tr+va),set(ids))
        self.assertEqual((tr,va),split_ids(ids))
    def test_actual_small_permeability_std_is_not_replaced_by_large_epsilon(self):
        mean,scale,constant=channel_stats(np.array([1e-14,3e-14]))
        self.assertFalse(constant);self.assertAlmostEqual(scale/1e-14,1.)
        np.testing.assert_allclose((np.array([1e-14,3e-14])-mean)/scale,[-1,1])
        mean,scale,constant=channel_stats(np.full(100,.2))
        self.assertTrue(constant);self.assertEqual(scale,1.)
    def test_temporal_sampling_matches_across_models_and_uses_all_realisations(self):
        a=temporal_indices(400,499,16,42,1);b=temporal_indices(400,499,16,42,1)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a.shape,(6400,2))
        self.assertEqual(len({tuple(x) for x in a}),6400)
        self.assertEqual(set(a[:,0]),set(range(400)))
        self.assertFalse(np.array_equal(a,temporal_indices(400,499,16,42,2)))
    def test_validation_values_do_not_enter_normalization(self):
        import tempfile,h5py
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'data.h5'
            with h5py.File(p,'w') as f:
                for name,values in [('train',[1.,3.]),('validation',[1e9,2e9])]:
                    g=f.create_group('data/'+name)
                    for key in ['permeability','porosity','pressure','saturation']:
                        g.create_dataset(key,data=values)
            stats=fit_normalization(p,['train'])
            self.assertEqual(stats['pressure']['mean'],2.)
            self.assertEqual(stats['pressure']['scale'],1.)
    def test_low_resolution_path_and_no_transposed_convolution(self):
        m=LowResolutionUNet(32).eval();sizes=[]
        m.enc1.register_forward_hook(lambda _,i,o:sizes.append(o.shape[-2:]))
        m.enc2.register_forward_hook(lambda _,i,o:sizes.append(o.shape[-2:]))
        with torch.no_grad():
            for size in [20,40,21]:
                y=m(torch.randn(2,32,size,size))
                self.assertEqual(tuple(y.shape),(2,32,size,size))
        self.assertEqual(sizes[:2],[torch.Size([10,10]),torch.Size([5,5])])
        self.assertFalse(any(isinstance(x,torch.nn.ConvTranspose2d) for x in m.modules()))
        self.assertTrue(any(isinstance(x,torch.nn.BatchNorm2d) for x in m.modules()))
    def test_original_and_lowres_share_exact_common_initial_parameters(self):
        a=build_model('U-FNO',42);b=build_model('U-FNO-LR',42)
        for k,v in a.state_dict().items():
            if '.unet' not in k:torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
        self.assertLess(sum(x.numel() for x in b.parameters()),sum(x.numel() for x in a.parameters()))
    def test_training_step_updates_weights_after_backward(self):
        m=torch.nn.Conv2d(4,2,1);o=torch.optim.AdamW(m.parameters(),lr=.01)
        before=m.weight.detach().clone()
        loss=training_step(m,o,torch.randn(2,4,4,4),torch.randn(2,2,4,4))
        self.assertTrue(np.isfinite(loss));self.assertFalse(torch.equal(before,m.weight.detach()))
    def test_seed_averaging_keeps100_independent_reservoir_units(self):
        from evaluate_controlled import conditional_summary
        a=np.stack([np.full((100,2),x) for x in [1.,2.,3.]])
        v=conditional_summary(a)
        np.testing.assert_allclose(v['mean'],[2.,2.])
        np.testing.assert_allclose(v['ci_low'],v['mean'])
        np.testing.assert_allclose(v['ci_high'],v['mean'])
    def test_nonfinite_and_exact_threshold_count_as_failures(self):
        from evaluate_controlled import failure_mask
        errors=np.zeros((4,3,2))
        errors[1,1,0]=100.;errors[2,2,1]=np.nan;errors[3,0,1]=1000.
        np.testing.assert_array_equal(failure_mask(errors),[False,True,True,False])
    def test_nonfinite_mean_is_not_estimated_from_survivors(self):
        from evaluate_controlled import conditional_summary
        a=np.ones((3,100,2));a[0,4,0]=np.inf
        result=conditional_summary(a)
        self.assertIsNone(result['mean'][0]);self.assertIsNone(result['ci_low'][0])
        self.assertEqual(result['mean'][1],1.)
    def test_provenance_rejects_changed_artifact_or_experiment_identity(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        import provenance
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);artifact=root/'config.json';artifact.write_text('{}')
            (root/'training_provenance.json').write_text(json.dumps({'identity':{'version':1},'artifacts':{'config.json':provenance.digest(artifact)}}))
            with patch.object(provenance,'HERE',root),patch.object(provenance,'identity',return_value={'version':1}):
                provenance.verify()
                artifact.write_text('{"changed":true}')
                with self.assertRaises(AssertionError):provenance.verify()
            artifact.write_text('{}')
            with patch.object(provenance,'HERE',root),patch.object(provenance,'identity',return_value={'version':2}):
                with self.assertRaises(AssertionError):provenance.verify()

if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
