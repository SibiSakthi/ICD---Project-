"""
    below code simulate the setting given in the main result and VCG mechanism and shows us that the payoff and positions are same
    """
import numpy as np
from typing import List, Tuple
import random

class GeneralizedEnglishAuction:
    def __init__(self, n_slots: int, ctrs: List[float], values: List[float]):
        """
        Initialize the Generalized English Auction.
        
        Parameters:
        - n_slots: Number of advertising slots
        - ctrs: Click-through rates for each slot (descending order)
        - values: Advertisers' private per-click values
        """
        self.n_slots = n_slots
        self.ctrs = ctrs
        self.values = values
        self.n_advertisers = len(values)
        
        # Ensure we have CTRs for all slots plus a zero CTR for no slot
        assert len(ctrs) == n_slots + 1, "CTRs should be provided for all slots plus a zero CTR"
        assert ctrs[-1] == 0, "Last CTR should be 0"
        
    def equilibrium_dropout_price(self, i: int, history: List[float], value: float) -> float:
        """
        Calculate the equilibrium dropout price according to the theorem.
        
        Parameters:
        - i: Number of advertisers remaining (including current one)
        - history: History of dropout prices
        - value: Advertiser's private per-click value
        
        Returns:
        - Dropout price
        """
        # Get bi-1 (most recent dropout price)
        if not history:
            bi_minus_1 = 0
        else:
            bi_minus_1 = history[-1]
        
        # With i advertisers remaining, the next to drop out gets position i-1
        position_index = i - 1
        
        # Get CTRs for current and next higher positions
        alpha_i = self.ctrs[position_index] if position_index<len(self.ctrs) else  self.ctrs[-2]
        alpha_i_minus_1 = self.ctrs[position_index - 1] if position_index<len(self.ctrs)-1 else  self.ctrs[-3]
        
        # Apply the formula in therom 2 : pk(i, h, sk) = sk - (αi/αi-1)(sk - bi-1)
        dropout_price = value - (alpha_i / alpha_i_minus_1) * (value - bi_minus_1)
        return dropout_price
    
    def run_auction(self) -> Tuple[List[int], List[float], List[float]]:
        """
        Run the Generalized English Auction.
        
        Returns:
        - allocation: List of advertiser indices assigned to each slot
        - prices: Per-click prices for each advertiser
        - history: Dropout price history
        """
        active_advertisers = list(range(self.n_advertisers))
        history = []  # Dropout price history
        allocation = [-1] * self.n_slots  # Which advertiser gets which slot
        prices = [0] * self.n_advertisers  # Per-click prices
        
        # Start with the lowest slot to be filled first
        next_slot_to_fill = self.n_slots - 1
        dropout_prices = []

        while len(active_advertisers) > self.n_slots:
            dropout_prices = []
            for adv in active_advertisers:
                price = self.equilibrium_dropout_price(
                    len(active_advertisers),
                    history,
                    self.values[adv]
                )
                dropout_prices.append((adv, price))
            # Find advertiser with lowest dropout price
            dropout_prices.sort(key=lambda x: x[1])
            next_dropout, price = dropout_prices[0]
            prices[next_dropout] = 0  # Dropped out, no slot, pays nothing
            history.append(price)
            active_advertisers.remove(next_dropout)
        # Auction continues until only one advertiser remains or all slots filled
        while len(active_advertisers) > 1 and next_slot_to_fill >= 0:
            # Calculate dropout prices for each active advertiser
            dropout_prices = []
            for adv in active_advertisers:
                price = self.equilibrium_dropout_price(
                    len(active_advertisers), 
                    history, 
                    self.values[adv]
                )
                dropout_prices.append((adv, price))
            
            # Find advertiser with lowest dropout price
            dropout_prices.sort(key=lambda x: x[1])
            next_dropout, price = dropout_prices[0]
            
            # Assign this advertiser to the next available slot
            allocation[next_slot_to_fill] = next_dropout
            
            # This advertiser pays their dropout price
            prices[next_dropout] = history[-1] if history else 0

            next_slot_to_fill -= 1
            
            # Update history and active advertisers
            history.append(price)
            active_advertisers.remove(next_dropout)
        
        # Last remaining advertiser gets the highest slot
        if active_advertisers and next_slot_to_fill >= 0:
            allocation[0] = active_advertisers[0]
            
            # The price for the top slot is the last dropout price
            if history:
                prices[active_advertisers[0]] = history[-1]
            else:
                prices[active_advertisers[0]] = 0
        print(dropout_prices,"dropout")
        return allocation, prices, history
    
    def display_results(self, allocation: List[int], prices: List[float]):
        """
        Display the auction results.
        """
        print("\nAuction Results:")
        print("----------------")
        print("Slot | CTR | Advertiser | Value | Price | Payment | Utility")
        print("-" * 70)
        
        total_revenue = 0
        
        for slot in range(self.n_slots):
            adv = allocation[slot]
            if adv < 0:
                print(f"{slot:4d} | {self.ctrs[slot]:.2f} | {'None':9s} | {'N/A':>5} | {'N/A':>5} | {'N/A':>7} | {'N/A':>7}")
                continue
            
            ctr = self.ctrs[slot]
            value = self.values[adv]
            price = prices[adv]
            payment = price * ctr
            utility = (value - price) * ctr
            
            print(f"{slot:4d} | {ctr:.2f} | {adv:9d} | {value:.2f} | {price:.2f} | {payment:.2f} | {utility:.2f}")
            
            total_revenue += payment
            
        print("-" * 70)
        print(f"Total Revenue: {total_revenue:.2f}")
        
    def calculate_vcg_payments(self) -> List[float]:
        """
        Calculate VCG payments for comparison.
        
        In a VCG auction for ad slots:
        - Slots are allocated to maximize total value (highest value gets highest CTR slot)
        - Each advertiser pays the externality they impose on others
        """
        # Sort advertisers by value
        sorted_indices = np.argsort(-np.array(self.values))
        
        vcg_prices = [0] * self.n_advertisers
        
        # Calculate VCG prices for advertisers who get slots
        for i in range(min(self.n_slots, len(sorted_indices))):
            advertiser = sorted_indices[i]
            
            # The VCG payment formula for position auctions:
            # sum over positions below i: (CTR_difference) * (value of advertiser who would get that position)
            payment = 0
            for j in range(i+1, min(self.n_slots + 1, len(sorted_indices))):
                next_advertiser = sorted_indices[j]
                payment += (self.ctrs[j-1] - self.ctrs[j]) * self.values[next_advertiser]
                
            # Convert to per-click price
            vcg_prices[advertiser] = payment / self.ctrs[i] if self.ctrs[i] > 0 else 0
        
        return vcg_prices

def run_simulation(n_slots=4, n_advertisers=4, seed=35):
    #np.random.seed(seed)
    
    # Generate CTRs - decreasing with position
    ctrs = [random.random() for _ in range(n_slots)]

    ctrs.sort(reverse=True)
    ctrs.append(0)  # Add zero CTR for no slot
    
    # Generate advertiser values from a gamma distribution
    values = np.random.gamma(shape=5, scale=2, size=n_advertisers)
    
    print("Auction Parameters:")
    print(f"Number of slots: {n_slots}")
    print(f"Number of advertisers: {n_advertisers}")
    print(f"CTRs: {ctrs}")
    print(f"Advertiser values: {values}")
    
    # Run the auction
    auction = GeneralizedEnglishAuction(n_slots, ctrs, values)
    allocation, prices, history = auction.run_auction()
    
    # Display results
    auction.display_results(allocation, prices)
    
    # Compare with VCG payments
    vcg_prices = auction.calculate_vcg_payments()
    
    print("\nComparison with VCG:")
    print("--------------------")
    print("Advertiser | Value | GSP Price | VCG Price")
    print("-" * 50)
    
    for adv in range(auction.n_advertisers):
        if adv in allocation:
            slot = allocation.index(adv)
            print(f"{adv:9d} | {auction.values[adv]:.2f} | {prices[adv]:.2f} | {vcg_prices[adv]:.2f}")
        else:
            print(f"{adv:9d} | {auction.values[adv]:.2f} | {'N/A':>8} | {vcg_prices[adv]:.2f}")
    
    return auction, allocation, prices, vcg_prices, history

# Run the simulation
auction, allocation, prices, vcg_prices, history = run_simulation()
