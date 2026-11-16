"""Account-wide card counting, consolidation and pending-admission races."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import gpu_account_policy as P


class AccountPolicyTests(unittest.TestCase):
    def setUp(self):
        self.settings = dict(preferred_gpu_count=5, max_gpu_count=6, account_uid=os.getuid())
        self.cards = [dict(uuid=f'GPU-{i}', index=i, free_mib=32000) for i in range(8)]

    def test_prefer_occupied_even_when_new_card_is_emptier(self):
        self.cards[7]['free_mib'] = 48000
        self.assertEqual(P.choose([self.cards[7], self.cards[1]], {'GPU-1'}, self.settings), [self.cards[1]])

    def test_sixth_card_available_when_existing_cards_cannot_fit(self):
        self.assertEqual(P.choose(self.cards[5:], {f'GPU-{i}' for i in range(5)}, self.settings), self.cards[5:])

    def test_seventh_card_blocked_and_existing_cards_still_usable(self):
        occupied = {f'GPU-{i}' for i in range(6)}
        self.assertEqual(P.choose(self.cards[6:], occupied, self.settings), [])
        self.assertEqual(P.choose([self.cards[1], self.cards[7]], occupied, self.settings), [self.cards[1]])

    def test_over_limit_drain_never_preempts_existing_jobs(self):
        occupied = {f'GPU-{i}' for i in range(7)}
        self.assertEqual(P.choose(self.cards, occupied, self.settings), self.cards[:7])

    def test_migration_frees_old_card_and_does_not_immediately_reacquire_it(self):
        occupied = {f'GPU-{i}' for i in (0, 1, 2, 4, 6, 7)}
        self.assertEqual(P.choose([self.cards[3], self.cards[4]], occupied, self.settings), [self.cards[4]])

    def test_maintenance_hold_stops_only_new_admissions(self):
        self.assertEqual(P.choose(self.cards, {'GPU-0'}, dict(self.settings, hold_new_workers=True)), [])

    def test_actual_foreign_processes_dead_leases_and_pid_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); leases = root/'status/checkpoint_gpu_leases'; leases.mkdir(parents=True)
            for pid, token, uuid in [(30, 'pending', 'GPU-2'), (40, 'old', 'GPU-3'), (50, 'dead', 'GPU-4')]:
                (leases/f'GPU-{pid}.json').write_text(json.dumps(dict(pid=pid, process_start_ticks=token, gpu=dict(uuid=uuid))))
            identities = {10:(os.getuid(),'a'), 20:(os.getuid()+1,'b'), 30:(os.getuid(),'pending'), 40:(os.getuid(),'reused')}
            with patch.object(P.subprocess,'check_output',return_value='GPU-0, 10\nGPU-1, 20\n'), patch.object(P,'process_identity',side_effect=identities.get):
                self.assertEqual(P.occupied_gpus(root), {'GPU-0','GPU-2'})

    def test_pending_claim_blocks_competing_seventh_card_before_cuda_starts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'configs').mkdir()
            (root/'configs/gpu-account-policy.json').write_text(json.dumps(self.settings))
            raw = ''.join(f'GPU-{i}, {100+i}\n' for i in range(5))
            identities = {100+i:(os.getuid(),str(i)) for i in range(5)}
            identities[200] = (os.getuid(),'pending')
            with patch.object(P.subprocess,'check_output',return_value=raw), patch.object(P,'process_identity',side_effect=identities.get):
                with P.admission_lock(root):
                    self.assertEqual(P.candidates(root,[self.cards[5]]),[self.cards[5]])
                    P.register(root,self.cards[5],200,12000,'test')
                with P.admission_lock(root):
                    self.assertEqual(P.candidates(root,[self.cards[6]]),[])

    def test_query_errors_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'configs').mkdir()
            (root/'configs/gpu-account-policy.json').write_text(json.dumps(self.settings))
            with patch.object(P.subprocess,'check_output',side_effect=OSError('NVML unavailable')):
                self.assertEqual(P.candidates(root,self.cards), [])
            self.assertIn('NVML unavailable', (root/f'status/gpu_account_policy/last_admission_{os.getpid()}.json').read_text())


if __name__ == '__main__':
    unittest.main()

class MemoryReservationTests(unittest.TestCase):
    def test_unallocated_reservations_count_once_and_ignore_stale_leases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); folder=root/'status/checkpoint_gpu_leases';folder.mkdir(parents=True)
            member=dict(pid=20,process_start_ticks='live',gpu=dict(uuid='GPU-0'),memory_reservation_mib=10000)
            (folder/'GPU-0.20.json').write_text(json.dumps(member))
            (folder/'GPU-0.json').write_text(json.dumps(dict(gpu=dict(uuid='GPU-0'),members=[member])))
            (folder/'GPU-0.30.json').write_text(json.dumps(dict(member,pid=30,process_start_ticks='stale')))
            with patch.object(P.subprocess,'check_output',return_value='GPU-0, 20, 3000\n'), patch.object(P,'process_identity',side_effect=lambda pid:(os.getuid(),'live')):
                devices=P.available_memory(root,[dict(uuid='GPU-0',free_mib=20000)])
            self.assertEqual(devices[0]['free_mib'],13000)
            self.assertEqual(devices[0]['pending_reservation_mib'],7000)

    def test_old_pending_lease_without_amount_defers_sharing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); folder=root/'status/checkpoint_gpu_leases';folder.mkdir(parents=True)
            (folder/'GPU-0.20.json').write_text(json.dumps(dict(pid=20,process_start_ticks='live',gpu=dict(uuid='GPU-0'))))
            with patch.object(P.subprocess,'check_output',return_value=''), patch.object(P,'process_identity',return_value=(os.getuid(),'live')):
                devices=P.available_memory(root,[dict(uuid='GPU-0',free_mib=20000)])
            self.assertEqual(devices[0]['free_mib'],0)

class WorkerAdmissionTests(unittest.TestCase):
    def test_spawn_and_registration_happen_under_same_lock(self):
        from contextlib import contextmanager
        state={'locked':False}; events=[]; gpu=dict(uuid='GPU-2',free_mib=24000)
        @contextmanager
        def lock(root):
            self.assertFalse(state['locked']);state['locked']=True
            yield
            state['locked']=False
        class Child:
            pid=12345
            def wait(self):
                self_outer.assertFalse(state['locked']);events.append('wait');return 0
        self_outer=self
        def spawn(*args,**kwargs):
            self.assertTrue(state['locked']);self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'GPU-2')
            events.append('spawn');return Child()
        def register(*args):
            self.assertTrue(state['locked']);events.append('register')
        with patch.object(P,'policy',return_value={'max_gpu_count':5}),patch.object(P,'admission_lock',lock),patch.object(P,'inventory',return_value=[gpu]),patch.object(P,'available_memory',side_effect=lambda root,cards:cards),patch.object(P,'candidates',side_effect=lambda root,cards:cards),patch.object(P.subprocess,'Popen',side_effect=spawn),patch.object(P,'register',side_effect=register):
            self.assertEqual(P.run_command('/tmp', ['python','worker.py'], 20000),0)
        self.assertEqual(events,['spawn','register','wait'])
