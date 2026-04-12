// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import {AccessControl} from "@openzeppelin/contracts/access/AccessControl.sol";

import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Capped.sol";
import "./LinearTokenVesting.sol";

/// @title ZEN official ERC-20 smart contract
/// @notice Minting role is granted in the constructor to the Vault Contracts, responsible for
///         restoring EON and Zend balances.

contract ZenToken is ERC20Capped, AccessControl {
    // Create a new role identifier for the minter role
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");

    uint256 internal constant TOTAL_ZEN_SUPPLY = 21_000_000;
    uint256 internal constant TOKEN_SIZE = 10 ** 18;

    address public horizenFoundationVested;

    uint8 numOfMinters;

    address public horizenDaoVested;


    error AddressParameterCantBeZero(string paramName);
    error CallerNotMinter(address caller);

    modifier canMint() {
        // Checks that the calling account has the minter role
        if (!hasRole(MINTER_ROLE, msg.sender)) {
            revert CallerNotMinter(msg.sender);
        }
        _;
    }

    /// @notice Smart contract constructor
    /// @param tokenName Name of the token
    /// @param tokenSymbol Ticker of the token
    /// @param _eonBackupContract Address of EON Vault contract
    /// @param _zendBackupContract Address of ZEND Vault contract
    /// @param _horizenFoundationVested Address who will receive the remaining portion of Zen reserved to the Foundation (with locking period)
    /// @param _horizenDaoVested Address who will receive the remaining portion of Zen reserved to the DAO (with locking period)
    constructor(
        string memory tokenName,
        string memory tokenSymbol,
        address _eonBackupContract,
        address _zendBackupContract,
        address _horizenFoundationVested,
        address _horizenDaoVested
    ) ERC20(tokenName, tokenSymbol) ERC20Capped(TOTAL_ZEN_SUPPLY * TOKEN_SIZE) {
        if (_eonBackupContract == address(0))
            revert AddressParameterCantBeZero("_eonBackupContract");
        if (_zendBackupContract == address(0))
            revert AddressParameterCantBeZero("_zendBackupContract");
        if (_horizenFoundationVested == address(0))
            revert AddressParameterCantBeZero("_horizenFoundationVested");
        if (_horizenDaoVested == address(0))
            revert AddressParameterCantBeZero("_horizenDaoVested");

        // Grant the minter role to a specified account
        _grantRole(MINTER_ROLE, _eonBackupContract);
        _grantRole(MINTER_ROLE, _zendBackupContract);
        numOfMinters = 2;
        
        horizenFoundationVested = _horizenFoundationVested;
        horizenDaoVested = _horizenDaoVested;
    }

    function mint(address to, uint256 amount) public canMint {
        _mint(to, amount);
    }

    function notifyMintingDone() public canMint {
        _revokeRole(MINTER_ROLE, msg.sender);
        unchecked {
            --numOfMinters;
        }
        if (numOfMinters == 0) {
            uint256 remainingSupply = cap() - totalSupply();
            //Horizen DAO is eligible of 60% of the remaining supply. The rest is for the Foundation.
            uint256 daoSupply = (remainingSupply * 6) / 10;
            uint256 foundationSupply = remainingSupply - daoSupply;

            uint256 daoInitialSupply = (daoSupply * 25) / 100;
            uint256 foundationInitialSupply = (foundationSupply * 25) / 100;
            _mint(
                LinearTokenVesting(horizenFoundationVested).beneficiary(),
                foundationInitialSupply
            );
            _mint(
                horizenFoundationVested,
                foundationSupply - foundationInitialSupply
            );
            _mint(
                LinearTokenVesting(horizenDaoVested).beneficiary(),
                daoInitialSupply
            );
            _mint(horizenDaoVested, daoSupply - daoInitialSupply);

            LinearTokenVesting(horizenFoundationVested).startVesting();
            LinearTokenVesting(horizenDaoVested).startVesting();
        }
    }
}
