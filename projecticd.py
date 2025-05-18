import numpy as np

def simulate_auctions(
    n_advertisers=5, 
    n_slots=3, 
    ctrs=None, 
    value_dist='uniform', 
    dist_params=(0, 1),
    n_trials=1,
    random_seed=None
):
    """
    Simulate VCG and GSP auctions for online ad slots.
    
    Parameters:
        n_advertisers: Number of advertisers
        n_slots: Number of ad slots
        ctrs: List of click-through rates for slots (descending order)
        value_dist: 'uniform' or 'normal'
        dist_params: Parameters for the value distribution
        n_trials: Number of simulation runs
        random_seed: Seed for reproducibility
        
    Returns:
        results: List of dictionaries with auction outcomes per trial
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    if ctrs is None:
        # Default: exponentially decreasing CTRs
        ctrs = [0.5**i for i in range(n_slots)]
    results = []
    for _ in range(n_trials):
        # Sample true values per click for each advertiser
        if value_dist == 'uniform':
            vals = np.random.uniform(dist_params[0], dist_params[1], n_advertisers)
        elif value_dist == 'normal':
            vals = np.random.normal(dist_params[0], dist_params[1], n_advertisers)
            vals = np.maximum(vals, 0)  # Ensure non-negative
        else:
            raise ValueError("Unsupported distribution")
        
        # Assume truthful bidding for both mechanisms (VCG is truthful, GSP for comparison)
        bids = vals.copy()
        # Rank advertisers by bid (descending)
        idx_sorted = np.argsort(-bids)
        sorted_bids = bids[idx_sorted]
        sorted_vals = vals[idx_sorted]
        
        # Assign slots
        slot_assign = idx_sorted[:n_slots]
        slot_vals = sorted_vals[:n_slots]
        slot_bids = sorted_bids[:n_slots]
        slot_ctrs = ctrs[:n_slots]
        
        # GSP payments: Each pays the next-highest bid per click
        gsp_payments = []
        for i in range(n_slots):
            if i+1 < n_advertisers:
                pay = sorted_bids[i+1]
            else:
                pay = 0.0
            gsp_payments.append(pay)
        gsp_payments = np.array(gsp_payments)
        gsp_revenue = np.sum(slot_ctrs * gsp_payments)
        gsp_utilities = slot_ctrs * (slot_vals - gsp_payments)
        
        # VCG payments: Externality imposed on others
        vcg_payments = []
        for i in range(n_slots):
            # Remove advertiser i and recompute total value for others
            others = np.delete(sorted_vals, i)
            # Assign slots to others
            others_sorted = np.sort(others)[::-1]
            # Value with i present
            value_with_i = np.sum(slot_ctrs * sorted_vals[:n_slots])
            # Value with i absent
            value_without_i = np.sum(slot_ctrs * others_sorted[:n_slots])
            # Payment is the difference in value to others, divided by CTR
            pay = (value_without_i - (value_with_i - slot_ctrs[i]*slot_vals[i])) / slot_ctrs[i]
            vcg_payments.append(pay)
        vcg_payments = np.array(vcg_payments)
        vcg_revenue = np.sum(slot_ctrs * vcg_payments)
        vcg_utilities = slot_ctrs * (slot_vals - vcg_payments)
        
        results.append({
            'vals': vals,
            'bids': bids,
            'slot_assign': slot_assign,
            'slot_ctrs': slot_ctrs,
            'gsp_payments': gsp_payments,
            'gsp_revenue': gsp_revenue,
            'gsp_utilities': gsp_utilities,
            'vcg_payments': vcg_payments,
            'vcg_revenue': vcg_revenue,
            'vcg_utilities': vcg_utilities,
        })
    return results

# Example usage:
if __name__ == "__main__":
    n_trials = 5
    results = simulate_auctions(
        n_advertisers=5, 
        n_slots=3, 
        ctrs=[0.8, 0.5, 0.3], 
        value_dist='uniform', 
        dist_params=(1, 10),
        n_trials=n_trials,
        random_seed=42
    )
    for i, res in enumerate(results):
        print(f"Trial {i+1}:")
        print("  Values:", np.round(res['vals'], 2))
        print("  GSP payments (per click):", np.round(res['gsp_payments'], 2))
        print("  VCG payments (per click):", np.round(res['vcg_payments'], 2))
        print("  GSP revenue:", np.round(res['gsp_revenue'], 2))
        print("  VCG revenue:", np.round(res['vcg_revenue'], 2))
        print()
