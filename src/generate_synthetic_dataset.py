"""
Enhanced Synthetic Dataset Generator for Fraud Detection
Features:
- One-way conversations (voicemails, automated messages)
- Varied speaker labels (agent/customer, scammer/victim, rep/client, etc.)
- Three-way conversations (scammer + accomplice + victim)
- 750+ edge cases for robust model training
"""

import csv
import os
import random
from typing import List, Tuple

import pandas as pd
import structlog
from faker import Faker

# Initialize Faker
fake = Faker()
logger = structlog.get_logger(__name__)

# ============================================================================
# SPEAKER LABEL VARIATIONS
# ============================================================================

# Different ways to label speakers in conversations
SPEAKER_PAIRS = [
    ("caller", "receiver"),
    ("agent", "customer"),
    ("scammer", "victim"),
    ("rep", "client"),
    ("operator", "user"),
    ("voice", "listener"),
    ("bot", "person"),
    ("system", "account_holder"),
]

# For three-way conversations
THREE_WAY_LABELS = [
    ("scammer", "accomplice", "victim"),
    ("agent1", "agent2", "customer"),
    ("caller", "manager", "receiver"),
    ("rep", "supervisor", "client"),
]

# ============================================================================
# ONE-WAY CONVERSATION TEMPLATES (Voicemails, Automated Messages)
# ============================================================================

oneway_fraud_templates = [
    # Automated voice messages
    (
        "system: This is an automated message from {organization}. "
        "Your {account_type} has been {action}. Press 1 to speak with an agent "
        "or call {phone_number} immediately to resolve this issue. "
        "Reference number {ref_number}."
    ),
    (
        "voice: Attention! This is a final notice from {agency}. "
        "There is a warrant for your arrest due to {threat}. "
        "Call {phone_number} within 24 hours to avoid legal action."
    ),
    (
        "bot: Hello, this is {company} security alert. "
        "We detected unauthorized access to your account from {location}. "
        "To secure your account, please verify your {sensitive_info} "
        "by calling {phone_number}."
    ),
    # Voicemail scams
    (
        "voicemail: Hi, this is {name} from {indian_bank}. "
        "Your account shows suspicious activity. We've temporarily blocked your card. "
        "Please call us back urgently at {phone_number} and provide your "
        "{indian_sensitive_info} to unlock it."
    ),
    (
        "voicemail: Sir, this is regarding your {indian_item}. "
        "There's a discrepancy in your records. You need to update your KYC immediately. "
        "Call {phone_number} and share the OTP we sent to complete verification."
    ),
    (
        "automated_msg: This is {indian_payment_app}. "
        "Your account has been debited Rs. {amount}. "
        "If you did not authorize this transaction, call {phone_number} immediately "
        "and provide the OTP to reverse it."
    ),
    # Prize/lottery voicemails
    (
        "message: Congratulations! You've won {indian_prize} in our lucky draw. "
        "To claim your prize, call {phone_number} and provide your "
        "{indian_sensitive_info} for verification. Offer expires in 24 hours."
    ),
    # Utility/service disconnection
    (
        "automated_call: This is {indian_company}. "
        "Your service will be disconnected due to pending payment of Rs. {amount}. "
        "Pay immediately by calling {phone_number} or your connection "
        "will be terminated today."
    ),
]

oneway_legit_templates = [
    "automated_msg: This is a reminder from {company} about your appointment on {date}. If you need to reschedule, please call us. Thank you.",
    "voicemail: Hi, this is {name} from {organization}. We're following up on your recent inquiry. Please call us back at your convenience. Thanks!",
    "system: Your {service} payment of ${amount} has been successfully processed. Thank you for your business.",
    "message: This is {company} customer service. Your order has been shipped and will arrive by {date}. Track your package online.",
]

# ============================================================================
# THREE-WAY CONVERSATION TEMPLATES
# ============================================================================

threeway_fraud_templates = [
    # Scammer + Accomplice + Victim
    (
        "scammer: Hello, this is {name} from {indian_bank} security. "
        "victim: Yes? scammer: Ma'am, we've detected unauthorized access to your account. "
        "I'm transferring you to my supervisor. "
        "accomplice: Hello ma'am, this is {name2}, senior security officer. "
        "We need to secure your account immediately. victim: What should I do? "
        "accomplice: Please share the OTP we just sent. "
        "scammer: Yes ma'am, quickly before the hacker accesses your account. "
        "victim: {indian_edge_response}"
    ),
    (
        "agent1: Sir, I'm calling from {organization} about your {account_type}. "
        "customer: Okay. agent1: Let me connect you with our verification department. "
        "agent2: Hello sir, this is {name} from verification. "
        "We need to confirm your {sensitive_info} to process your request. "
        "customer: {edge_response}"
    ),
    # Bank scam with fake manager
    (
        "caller: Good morning, this is {name} from {indian_bank}. "
        "receiver: Yes? caller: Sir, there's an issue with your account. "
        "Let me connect you to my manager. "
        "manager: Hello sir, I'm {name2}, branch manager. "
        "Your account has been flagged for {indian_threat}. "
        "We need your {indian_sensitive_info} to clear this. "
        "receiver: {indian_edge_response}"
    ),
    # Tech support scam with escalation
    (
        "rep: This is {company} technical support. Your computer has a virus. "
        "client: Really? rep: Yes, let me transfer you to our senior technician. "
        "supervisor: Hello, this is {name}. We need remote access to fix this. "
        "Please provide your {sensitive_info} to authorize the connection. "
        "client: {edge_response}"
    ),
    # Investment scam with multiple agents
    (
        "agent1: Sir, this is {name} from {company} investment services. "
        "We have an exclusive opportunity for you. customer: What is it? "
        "agent1: Let me connect you with our senior advisor. "
        "agent2: Hello sir, I'm {name2}. "
        "We're offering guaranteed returns of 40% per month. "
        "Just transfer Rs. {amount} to get started. "
        "Provide your {indian_sensitive_info} for account setup. "
        "customer: {indian_edge_response}"
    ),
]

threeway_legit_templates = [
    (
        "agent1: Hi, this is {name} from {company} customer service. "
        "customer: Hello. agent1: Let me transfer you to our specialist for your inquiry. "
        "agent2: Hi, I'm {name2}. How can I help you today? "
        "customer: I have a question about my account. "
        "agent2: Sure, I'd be happy to assist."
    ),
    (
        "rep: This is {name} from {organization}. client: Yes? "
        "rep: I'm connecting you with our appointment coordinator. "
        "coordinator: Hello, this is {name2}. "
        "I'm calling to confirm your appointment on {date}. "
        "client: Yes, that works for me."
    ),
]

# ============================================================================
# EXPANDED EDGE CASE TEMPLATES (750+ variations)
# ============================================================================

# Category 1: Polite Professional Scams (100 templates)
polite_professional_scams = [
    (
        "{s1}: Good {time_of_day}, this is {name} from {legitimate_org}. "
        "I hope I'm not disturbing you. {s2}: No, it's fine. "
        "{s1}: We're conducting a routine security audit and noticed your account "
        "was accessed from {location}. For your protection, we'd like to verify "
        "a few details. {s2}: {edge_response}"
    ),
    (
        "{s1}: Hello, {name} speaking from {organization}. How are you today? "
        "{s2}: I'm fine, thanks. {s1}: Wonderful! I'm calling about a security matter. "
        "We noticed some unusual activity and want to ensure your account is safe. "
        "May I verify your {sensitive_info}? {s2}: {edge_response}"
    ),
    (
        "{s1}: Hi there, this is {name} with {company}. "
        "I hope you're having a great day. {s2}: Thank you. "
        "{s1}: I'm reaching out because we value your security. "
        "We've implemented new protection measures and need to update your account. "
        "Could you confirm your {sensitive_info}? {s2}: {edge_response}"
    ),
    (
        "{s1}: Good {time_of_day}, my name is {name} from {indian_bank}. "
        "How may I address you? {s2}: This is fine. "
        "{s1}: Thank you for your time. We're upgrading our security systems "
        "and need to verify account holders. May I have your {indian_sensitive_info} please? "
        "{s2}: {indian_edge_response}"
    ),
    (
        "{s1}: Hello, {name} here from {legitimate_org}. I apologize for the interruption. "
        "{s2}: It's okay. {s1}: Thank you for understanding. "
        "We're conducting mandatory verification as per new regulations. "
        "This will only take a moment. Your {sensitive_info}, please? {s2}: {edge_response}"
    ),
]

# Generate more variations
for i in range(95):
    polite_professional_scams.append(
        "{s1}: "
        + random.choice(
            [
                ("Good {time_of_day}, this is {name} from {organization}. " "I hope this is a good time?"),
                "Hello, {name} speaking from {company}. How are you doing today?",
                ("Hi, this is {name} with {legitimate_org}. " "I hope I'm not catching you at a bad time?"),
                ("Good {time_of_day}, my name is {name} from {indian_bank}. " "Is this a convenient time to talk?"),
            ]
        )
        + " {s2}: "
        + random.choice(
            [
                "Yes, go ahead.",
                "I have a few minutes.",
                "What's this about?",
                "Sure, what do you need?",
            ]
        )
        + " {s1}: "
        + random.choice(
            [
                ("Wonderful. We're conducting a security review and need to " "verify your {sensitive_info}."),
                (
                    "Great. We've noticed some activity on your account and need to "
                    "confirm your {sensitive_info} for security."
                ),
                ("Perfect. We're updating our records and need to verify " "your {indian_sensitive_info}."),
                ("Thank you. We need to confirm your identity. " "Could you provide your {sensitive_info}?"),
            ]
        )
        + " {s2}: {edge_response}"
    )

# Category 2: Trust-Building Scams (100 templates)
trust_building_scams = [
    (
        "{s1}: Hi, this is {name} from {company} customer loyalty program. "
        "{s2}: Hello. {s1}: You've been a valued customer for {years} years, "
        "and we wanted to personally thank you. {s2}: That's nice. "
        "{s1}: We're offering you an exclusive reward. To activate it, "
        "I just need to verify your {sensitive_info}. {s2}: {edge_response}"
    ),
    (
        "{s1}: Good {time_of_day}, {name} calling from {organization} VIP services. "
        "{s2}: Yes? {s1}: Congratulations! You've been selected for our premium "
        "membership program based on your excellent payment history. {s2}: Really? "
        "{s1}: Absolutely! To enroll you, I need to confirm your {sensitive_info}. "
        "{s2}: {edge_response}"
    ),
    (
        "{s1}: Hello, this is {name} from {indian_bank} relationship management. "
        "{s2}: Hi. {s1}: Sir, you're one of our most valued customers. "
        "We'd like to offer you a complimentary upgrade to premium banking. "
        "{s2}: What does that include? {s1}: Many benefits! To activate, "
        "please share your {indian_sensitive_info}. {s2}: {indian_edge_response}"
    ),
]

# Generate more variations
for i in range(97):
    trust_building_scams.append(
        "{s1}: "
        + random.choice(
            [
                "Hi, this is {name} from {company} rewards program.",
                "Good {time_of_day}, {name} from {organization} customer appreciation team.",
                "Hello, this is {name} with {indian_bank} privilege banking.",
                "Hi there, {name} calling from {company} loyalty services.",
            ]
        )
        + " {s2}: "
        + random.choice(
            [
                "Hello.",
                "Yes?",
                "Hi.",
                "What's this regarding?",
            ]
        )
        + " {s1}: "
        + random.choice(
            [
                "You've been a loyal customer for {years} years. We want to reward you with {offer}.",
                "Based on your excellent history, you qualify for our exclusive {legitimate_service} program.",
                "Congratulations! You're eligible for a special {offer} as a valued customer.",
                "We're offering you premium benefits at no cost. You've earned this!",
            ]
        )
        + " {s2}: "
        + random.choice(
            [
                "That sounds good.",
                "What do I need to do?",
                "Tell me more.",
                "Interesting.",
            ]
        )
        + " {s1}: "
        + random.choice(
            [
                "Just verify your {sensitive_info} to activate your rewards.",
                "I need to confirm your {indian_sensitive_info} to process this.",
                "Simply provide your {sensitive_info} and we'll set everything up.",
                "Let me verify your {indian_sensitive_info} to complete enrollment.",
            ]
        )
        + " {s2}: {edge_response}"
    )

# Category 3: Technical Jargon Scams (100 templates)
technical_jargon_scams = [
    (
        "{s1}: This is {name} from {company} technical security team. "
        "We've detected a {technical_term} vulnerability in your account. "
        "{s2}: What does that mean? {s1}: It means your data encryption has been compromised. "
        "We need to implement a security patch immediately. "
        "Please provide your {sensitive_info} to authorize the update. {s2}: {edge_response}"
    ),
    (
        "{s1}: Hello, {name} from {organization} cybersecurity division. "
        "Your account shows signs of a {technical_term} breach. {s2}: Is that serious? "
        "{s1}: Very serious. We need to reset your authentication protocols. "
        "Confirm your {sensitive_info} to proceed with the security upgrade. {s2}: {edge_response}"
    ),
]

# Generate more variations
for i in range(98):
    technical_jargon_scams.append(
        "{s1}: "
        + random.choice(
            [
                "This is {name} from {company} IT security.",
                "Hello, {name} with {organization} cyber defense team.",
                "Hi, this is {name} from {company} network security.",
                "{name} calling from {organization} data protection unit.",
            ]
        )
        + " "
        + random.choice(
            [
                "We've detected a {technical_term} vulnerability in your system.",
                "Your account shows a {technical_term} security breach.",
                "We've identified a {technical_term} exploit targeting your account.",
                "There's a critical {technical_term} issue that needs immediate attention.",
            ]
        )
        + " {s2}: "
        + random.choice(
            [
                "What should I do?",
                "Is this serious?",
                "What does that mean?",
                "How did this happen?",
            ]
        )
        + " {s1}: "
        + random.choice(
            [
                ("We need to patch your security immediately. " "Provide your {sensitive_info} to authorize."),
                ("Your {technical_term} needs reconfiguration. " "Confirm your {sensitive_info}."),
                ("We must update your authentication. " "Please verify your {indian_sensitive_info}."),
                ("Immediate action required. " "Share your {sensitive_info} to secure your account."),
            ]
        )
        + " {s2}: {edge_response}"
    )

# Category 4: Partial Truth Scams (100 templates)
partial_truth_scams = [
    (
        "{s1}: Hello, this is {name} calling about your recent {legitimate_service} inquiry. "
        "{s2}: Yes, I did inquire about that. {s1}: Great! To proceed with your application, "
        "I need to verify your identity. Can you confirm the {sensitive_info} on file? "
        "{s2}: {edge_response}"
    ),
    (
        "{s1}: Hi, {name} from {company}. You recently browsed our {legitimate_service} options online. "
        "{s2}: Yes, I was looking at that. {s1}: Perfect! I can help you complete the process. "
        "I just need your {sensitive_info} to pull up your information. {s2}: {edge_response}"
    ),
]

# Generate more variations
for i in range(98):
    partial_truth_scams.append(
        "{s1}: "
        + random.choice(
            [
                "Hello, this is {name} regarding your {legitimate_service} request.",
                "Hi, {name} calling about your recent {legitimate_service} inquiry.",
                "Good {time_of_day}, {name} from {company}. You contacted us about {legitimate_service}.",
                "This is {name}. You recently applied for {legitimate_service}.",
            ]
        )
        + " {s2}: "
        + random.choice(
            [
                "Yes, I did.",
                "That's correct.",
                "I remember.",
                "Yes, what about it?",
            ]
        )
        + " {s1}: "
        + random.choice(
            [
                "Excellent! To proceed, I need to verify your {sensitive_info}.",
                "Great! Let me pull up your file. What's your {sensitive_info}?",
                "Perfect! I just need to confirm your {indian_sensitive_info} to continue.",
                "Wonderful! For security, please provide your {sensitive_info}.",
            ]
        )
        + " {s2}: {edge_response}"
    )

# Category 5: Urgency Without Threats (100 templates)
urgency_without_threats_scams = [
    (
        "{s1}: Hello, this is {name} from {organization}. "
        "We're calling to inform you about an important deadline tomorrow. "
        "{s2}: What deadline? {s1}: Your {account_type} requires immediate verification "
        "to avoid service interruption. It's just a formality. "
        "Can you confirm your {sensitive_info}? {s2}: {edge_response}"
    ),
    (
        "{s1}: Hi, {name} with {company}. We have a time-sensitive matter regarding your account. "
        "{s2}: What is it? {s1}: Your {account_type} needs updating by end of day. "
        "Quick verification needed. What's your {sensitive_info}? {s2}: {edge_response}"
    ),
]

# Generate more variations
for i in range(98):
    urgency_without_threats_scams.append(
        "{s1}: "
        + random.choice(
            [
                "Hello, {name} from {organization}. Time-sensitive matter regarding your account.",
                "Hi, this is {name}. Important deadline approaching for your {account_type}.",
                "Good {time_of_day}, {name} calling. Urgent update needed for your account.",
                "This is {name} from {company}. Quick action required on your {account_type}.",
            ]
        )
        + " {s2}: "
        + random.choice(
            [
                "What's this about?",
                "What deadline?",
                "What do you need?",
                "Is something wrong?",
            ]
        )
        + " {s1}: "
        + random.choice(
            [
                "Just a routine update needed today. Verify your {sensitive_info} please.",
                ("Standard procedure, but time-sensitive. " "Confirm your {sensitive_info}."),
                "Quick verification required. What's your {indian_sensitive_info}?",
                ("Simple formality, but urgent. " "Please provide your {sensitive_info}."),
            ]
        )
        + " {s2}: {edge_response}"
    )

# Category 6: Indian Premium Banking Scams (100 templates)
indian_premium_scams = [
    (
        "{s1}: Namaste, this is {name} from {indian_bank} premium banking. "
        "{s2}: Hello. {s1}: Sir, as a valued customer, we're upgrading your account "
        "to premium status with zero charges. {s2}: That sounds good. "
        "{s1}: To activate, I just need to verify your {indian_sensitive_info}. "
        "This is a limited time offer. {s2}: {indian_edge_response}"
    ),
    (
        "{s1}: Good morning sir, {name} from {indian_bank} wealth management. "
        "{s2}: Yes? {s1}: We're offering you complimentary premium services "
        "worth Rs. 50,000 annually. {s2}: Really? {s1}: Yes sir! "
        "Just confirm your {indian_sensitive_info} to activate. {s2}: {indian_edge_response}"
    ),
]

# Generate more variations
for i in range(98):
    indian_premium_scams.append(
        "{s1}: "
        + random.choice(
            [
                "Namaste, this is {name} from {indian_bank} premium services.",
                "Good {time_of_day} sir, {name} calling from {indian_bank} privilege banking.",
                "Hello sir, {name} from {indian_bank} wealth management.",
                "Namaskar, this is {name} with {indian_bank} VIP services.",
            ]
        )
        + " {s2}: "
        + random.choice(
            [
                "Hello.",
                "Yes?",
                "Good morning.",
                "Tell me.",
            ]
        )
        + " {s1}: "
        + random.choice(
            [
                "Sir, we're upgrading your account to premium with zero charges.",
                "You're eligible for complimentary premium banking services.",
                "We're offering you exclusive benefits at no cost.",
                "Free upgrade to our platinum account for valued customers like you.",
            ]
        )
        + " {s2}: "
        + random.choice(
            [
                "That sounds good.",
                "What's included?",
                "How does this work?",
                "Tell me more.",
            ]
        )
        + " {s1}: "
        + random.choice(
            [
                "Just verify your {indian_sensitive_info} to activate. Limited time offer!",
                ("Simply confirm your {indian_sensitive_info} " "and we'll upgrade immediately."),
                ("Provide your {indian_sensitive_info} " "to complete the upgrade process."),
                ("Share your {indian_sensitive_info} " "to activate these premium benefits."),
            ]
        )
        + " {s2}: {indian_edge_response}"
    )

# Category 7: Compliance/Regulatory Scams (100 templates)
compliance_scams = [
    (
        "{s1}: Hello, I'm {name} from {indian_organization} compliance department. "
        "{s2}: Yes? {s1}: Sir, your {indian_item} is due for mandatory annual verification "
        "as per new government regulations. {s2}: I wasn't aware of this. "
        "{s1}: Yes sir, it's a new rule from this year. "
        "Please provide your {indian_sensitive_info} for verification. "
        "{s2}: {indian_edge_response}"
    ),
    (
        "{s1}: Good {time_of_day}, {name} from {organization} regulatory compliance. "
        "{s2}: Hello. {s1}: We're implementing new federal requirements. "
        "All account holders must verify their {sensitive_info} by end of month. "
        "{s2}: {edge_response}"
    ),
]

# Generate more variations
for i in range(98):
    compliance_scams.append(
        "{s1}: "
        + random.choice(
            [
                "Hello, {name} from {indian_organization} compliance division.",
                "Good {time_of_day}, {name} with {organization} regulatory department.",
                "This is {name} from {indian_organization} verification unit.",
                "Hi, {name} calling from {organization} compliance office.",
            ]
        )
        + " {s2}: "
        + random.choice(
            [
                "Yes?",
                "Hello.",
                "What's this about?",
                "Go ahead.",
            ]
        )
        + " {s1}: "
        + random.choice(
            [
                "New government regulations require mandatory {indian_item} verification.",
                "As per updated compliance rules, we need to verify your {account_type}.",
                "Recent regulatory changes mandate immediate verification of your {indian_item}.",
                "New federal requirements: all customers must update their {account_type}.",
            ]
        )
        + " {s2}: "
        + random.choice(
            [
                "I wasn't aware of this.",
                "When did this change?",
                "Is this mandatory?",
                "What do I need to do?",
            ]
        )
        + " {s1}: "
        + random.choice(
            [
                ("Yes sir, new rule. Please provide your {indian_sensitive_info} " "for compliance."),
                ("Mandatory as of this month. " "Verify your {sensitive_info} to avoid penalties."),
                ("Required by law. Share your {indian_sensitive_info} " "to complete verification."),
                "Government mandate. Confirm your {sensitive_info} immediately.",
            ]
        )
        + " {s2}: {indian_edge_response}"
    )

# Category 8: Code-Switching Scams (Hinglish) (50 templates)
code_switching_scams = [
    (
        "{s1}: Hello sir, myself {name} from {indian_bank}. {s2}: Yes, tell me. "
        "{s1}: Sir, actually your account mein kuch issue hai. We need to verify. "
        "{s2}: What issue? {s1}: Sir, for security purpose, please share the OTP. "
        "It's just a verification. {s2}: {indian_edge_response}"
    ),
    (
        "{s1}: Namaste sir, {name} bol raha hoon {indian_bank} se. {s2}: Haan? "
        "{s1}: Sir, aapke account mein suspicious activity detect hui hai. "
        "{s2}: Kya hua? {s1}: Sir, aapko OTP share karna hoga for verification. "
        "{s2}: {indian_edge_response}"
    ),
]

# Generate more variations
for i in range(48):
    code_switching_scams.append(
        "{s1}: "
        + random.choice(
            [
                "Hello sir, myself {name} from {indian_bank}.",
                "Namaste, {name} speaking from {indian_payment_app}.",
                "Good morning sir, {name} bol raha hoon {indian_bank} se.",
                "Sir, this is {name} calling from {indian_organization}.",
            ]
        )
        + " {s2}: "
        + random.choice(
            [
                "Yes, tell me.",
                "Haan, bolo.",
                "What is it?",
                "Yes?",
            ]
        )
        + " {s1}: "
        + random.choice(
            [
                "Sir, actually your account mein problem hai. Verification chahiye.",
                "Aapke {indian_item} mein discrepancy hai. OTP share karo.",
                "Sir, your account has been blocked. Aapko OTP dena hoga.",
                "Actually sir, security issue hai. Please provide OTP.",
            ]
        )
        + " {s2}: "
        + random.choice(
            [
                "What issue?",
                "Kya problem hai?",
                "Why?",
                "Tell me clearly.",
            ]
        )
        + " {s1}: "
        + random.choice(
            [
                "Sir, bas OTP share karo. It's for security only.",
                "Aapka {indian_sensitive_info} verify karna hai. Quick process.",
                "Sir, OTP bhejo. Account unlock ho jayega.",
                "Just verification sir. Please share the OTP we sent.",
            ]
        )
        + " {s2}: {indian_edge_response}"
    )

# Combine all edge case templates
all_edge_case_templates = (
    polite_professional_scams
    + trust_building_scams
    + technical_jargon_scams
    + partial_truth_scams
    + urgency_without_threats_scams
    + indian_premium_scams
    + compliance_scams
    + code_switching_scams
)

logger.info("Total edge case templates generated", count=len(all_edge_case_templates))

# ============================================================================
# DATA VALUES
# ============================================================================

# US-based
organizations = [
    "the Social Security Administration",
    "the IRS",
    "Medicare Services",
    "the Federal Trade Commission",
    "your bank",
    "the fraud department",
    "customer service",
    "technical support",
]
agencies = ["the Social Security Administration", "the FBI", "the Treasury Department", "Homeland Security"]
companies = ["Amazon", "Microsoft", "Apple Support", "Google", "your credit card company", "the billing department"]
departments = [
    "Fraud Prevention Department",
    "Account Security Division",
    "Customer Protection Unit",
    "Identity Verification Department",
]
account_types = ["social security number", "bank account", "credit card", "Medicare account", "tax records"]
items = ["social security number", "credit card", "identity", "account", "personal information"]
actions = ["suspended", "compromised", "flagged for fraud", "frozen", "deactivated"]
threats = [
    "Someone is using your information fraudulently",
    "Your account has been hacked",
    "We've detected unauthorized charges",
    "There's a warrant for your arrest",
    "Your benefits will be terminated",
]
sensitive_info = ["social security number", "bank account number", "credit card number", "date of birth", "PIN number"]
requests = ["verify your information", "confirm your identity", "provide your account details", "make a payment"]
consequences = ["suspended", "terminated", "frozen", "closed permanently"]
services = ["internet service", "phone service", "subscription", "warranty", "insurance policy"]
offers = ["a refund", "a special discount", "an upgrade", "compensation", "a prize"]
events = ["delivery", "service appointment", "renewal date", "payment due date"]

# India-based
indian_banks = [
    "State Bank of India",
    "HDFC Bank",
    "ICICI Bank",
    "Axis Bank",
    "Punjab National Bank",
    "Bank of Baroda",
    "Canara Bank",
    "Union Bank",
    "Kotak Mahindra Bank",
    "Yes Bank",
]
indian_payment_apps = ["Paytm", "PhonePe", "Google Pay", "BHIM UPI", "Amazon Pay", "MobiKwik"]
indian_organizations = ["UIDAI (Aadhaar)", "Income Tax Department", "EPFO", "Telecom Department", "RBI"]
indian_companies = ["Flipkart", "Amazon India", "Reliance Jio", "Airtel", "BSNL", "Vodafone Idea"]
indian_items = ["Aadhaar card", "PAN card", "bank account", "UPI ID", "debit card", "credit card"]
indian_accounts = ["bank account", "Aadhaar", "PAN card", "UPI account", "mobile wallet"]
indian_sensitive_info = [
    "Aadhaar number",
    "PAN card number",
    "UPI PIN",
    "debit card CVV",
    "net banking password",
    "ATM PIN",
    "OTP",
    "account number and IFSC code",
]
indian_threats = [
    "money laundering",
    "illegal transactions",
    "tax evasion",
    "fraudulent activities",
    "unauthorized access",
    "identity theft",
]
indian_prizes = ["Rs. 25 lakhs", "a new car", "Rs. 10 lakhs cashback", "iPhone 15 Pro", "gold coins worth Rs. 5 lakhs"]
amounts = ["50,000", "1,25,000", "2,500", "15,000", "75,000", "3,000", "25,000", "5,000", "10,000", "99,999"]

# Edge case specific
legitimate_org = [
    "your bank's security team",
    "customer service",
    "account verification department",
    "quality assurance team",
]
legitimate_service = ["loan", "credit card", "insurance", "investment", "account upgrade"]
technical_terms = [
    "SSL certificate",
    "two-factor authentication",
    "encryption key",
    "API token",
    "security protocol",
    "firewall",
    "VPN",
    "authentication token",
    "digital signature",
    "cryptographic hash",
]
time_of_day = ["morning", "afternoon", "evening"]
phone_numbers = ["1-800-" + str(random.randint(100, 999)) + "-" + str(random.randint(1000, 9999)) for _ in range(20)]
ref_numbers = ["REF" + str(random.randint(100000, 999999)) for _ in range(20)]
years = ["2", "3", "5", "7", "10"]
locations = ["New York", "California", "Texas", "Mumbai", "Delhi", "Bangalore", "London", "Dubai"]

# ============================================================================
# RESPONSE TEMPLATES
# ============================================================================

fraud_responses = [
    "I'm not comfortable giving that information over the phone.",
    "How do I know you're legitimate?",
    "Can I call you back to verify?",
    "That sounds suspicious.",
    "Okay, what do you need?",
    "I'm going to hang up and call the official number.",
    "Let me think about it.",
    "Can you give me your badge number?",
    "I'll report this to the authorities.",
]

indian_responses = [
    "I don't share OTP with anyone.",
    "Let me verify this with my bank first.",
    "I'll visit the branch directly.",
    "This sounds like a scam.",
    "Okay sir, what is the OTP number?",
    "I'm going to call the official customer care number.",
    "Why do you need my Aadhaar number?",
    "Can you give me your employee ID?",
    "I'll report this to cyber crime.",
    "My bank never asks for OTP on call.",
]

edge_responses = [
    "Let me check my records first.",
    "Can you send me an email about this?",
    "I'll need to speak with my spouse about this.",
    "Sure, let me get that information.",
    "I'm not sure I have that information handy.",
    "Can we do this later? I'm busy right now.",
    "Okay, I trust you. What do you need?",
    "I'll call back from the number on my card.",
    "Can you verify your identity first?",
    "This seems legitimate. Go ahead.",
]

indian_edge_responses = [
    "Okay sir, but can you tell me your employee code?",
    "Let me check with my family member first.",
    "I'll come to the branch tomorrow.",
    "Sure sir, what is the process?",
    "Can you send SMS confirmation?",
    "I'll verify on the app first.",
    "Okay, I'll share the OTP.",
    "Let me call the customer care number on the website.",
    "This sounds genuine. Please proceed.",
    "I'm not comfortable with this.",
]

legit_responses = [
    "Thank you for calling.",
    "I appreciate the information.",
    "That's helpful, thanks.",
    "Okay, I understand.",
    "Great, thank you.",
    "I'll take care of that.",
    "Thanks for the reminder.",
]

# ============================================================================
# GENERATION FUNCTIONS
# ============================================================================


def get_speaker_labels(conversation_type: str = "two_way") -> tuple:
    """Get appropriate speaker labels based on conversation type"""
    if conversation_type == "one_way":
        return random.choice(["system", "voice", "bot", "voicemail", "automated_msg", "message", "automated_call"])
    elif conversation_type == "three_way":
        return random.choice(THREE_WAY_LABELS)
    else:  # two_way
        return random.choice(SPEAKER_PAIRS)


def generate_oneway_fraud() -> Tuple[str, int]:
    """Generate a one-way fraudulent message (voicemail/automated)"""
    template = random.choice(oneway_fraud_templates)

    text = template.format(
        organization=random.choice(organizations),
        agency=random.choice(agencies),
        company=random.choice(companies),
        account_type=random.choice(account_types),
        action=random.choice(actions),
        threat=random.choice(threats),
        sensitive_info=random.choice(sensitive_info),
        phone_number=random.choice(phone_numbers),
        ref_number=random.choice(ref_numbers),
        indian_bank=random.choice(indian_banks),
        indian_payment_app=random.choice(indian_payment_apps),
        indian_company=random.choice(indian_companies),
        indian_item=random.choice(indian_items),
        indian_sensitive_info=random.choice(indian_sensitive_info),
        indian_prize=random.choice(indian_prizes),
        amount=random.choice(amounts),
        location=random.choice(locations),
        name=fake.last_name(),
    )

    return text, 1


def generate_oneway_legit() -> Tuple[str, int]:
    """Generate a one-way legitimate message"""
    template = random.choice(oneway_legit_templates)

    text = template.format(
        company=random.choice(["your doctor's office", "the pharmacy", "your insurance company"]),
        organization=random.choice(["your local clinic", "the community center"]),
        service=random.choice(services),
        date=fake.date_this_month().strftime("%B %d"),
        amount=random.choice(amounts),
        name=fake.first_name(),
    )

    return text, 0


def generate_threeway_fraud() -> Tuple[str, int]:
    """Generate a three-way fraudulent conversation"""
    template = random.choice(threeway_fraud_templates)
    labels = get_speaker_labels("three_way")

    # Replace generic labels with specific ones
    text = template
    for i, label in enumerate(labels):
        text = text.replace(["scammer", "agent1", "caller", "rep"][min(i, 3)], label)

    text = text.format(
        name=fake.last_name(),
        name2=fake.last_name(),
        organization=random.choice(organizations),
        company=random.choice(companies),
        indian_bank=random.choice(indian_banks),
        account_type=random.choice(account_types),
        sensitive_info=random.choice(sensitive_info),
        indian_sensitive_info=random.choice(indian_sensitive_info),
        indian_threat=random.choice(indian_threats),
        amount=random.choice(amounts),
        edge_response=random.choice(edge_responses),
        indian_edge_response=random.choice(indian_edge_responses),
    )

    return text, 1


def generate_threeway_legit() -> Tuple[str, int]:
    """Generate a three-way legitimate conversation"""
    template = random.choice(threeway_legit_templates)
    labels = get_speaker_labels("three_way")

    text = template
    for i, label in enumerate(labels):
        text = text.replace(["agent1", "rep"][min(i, 1)], label)

    text = text.format(
        name=fake.first_name(),
        name2=fake.first_name(),
        company=random.choice(["your doctor's office", "the pharmacy"]),
        organization=random.choice(["your local clinic", "the community center"]),
        date=fake.date_this_month().strftime("%B %d"),
    )

    return text, 0


def generate_edge_case() -> Tuple[str, int]:
    """Generate an edge case fraud conversation"""
    template = random.choice(all_edge_case_templates)
    labels = get_speaker_labels("two_way")

    # Replace {s1} and {s2} with actual speaker labels
    text = template.replace("{s1}", labels[0]).replace("{s2}", labels[1])

    text = text.format(
        name=fake.last_name(),
        name2=fake.last_name(),
        time_of_day=random.choice(time_of_day),
        legitimate_org=random.choice(legitimate_org),
        legitimate_service=random.choice(legitimate_service),
        company=random.choice(companies + indian_companies),
        organization=random.choice(organizations),
        indian_bank=random.choice(indian_banks),
        indian_payment_app=random.choice(indian_payment_apps),
        indian_organization=random.choice(indian_organizations),
        technical_term=random.choice(technical_terms),
        account_type=random.choice(account_types),
        indian_item=random.choice(indian_items),
        sensitive_info=random.choice(sensitive_info),
        indian_sensitive_info=random.choice(indian_sensitive_info),
        years=random.choice(years),
        location=random.choice(locations),
        offer=random.choice(offers),
        edge_response=random.choice(edge_responses),
        indian_edge_response=random.choice(indian_edge_responses),
    )

    return text, 1


def generate_fraud_call() -> Tuple[str, int]:
    """Generate a fraudulent call (various types)"""
    # 10% one-way, 10% three-way, 80% edge cases (two-way)
    conv_type = random.choices(["oneway", "threeway", "edge"], weights=[0.1, 0.1, 0.8])[0]

    if conv_type == "oneway":
        return generate_oneway_fraud()
    elif conv_type == "threeway":
        return generate_threeway_fraud()
    else:
        return generate_edge_case()


def generate_legit_call() -> Tuple[str, int]:
    """Generate a legitimate call (various types)"""
    # 20% one-way, 10% three-way, 70% regular two-way
    conv_type = random.choices(["oneway", "threeway", "regular"], weights=[0.2, 0.1, 0.7])[0]

    if conv_type == "oneway":
        return generate_oneway_legit()
    elif conv_type == "threeway":
        return generate_threeway_legit()
    else:
        # Regular two-way legitimate conversation
        labels = get_speaker_labels("two_way")
        template = random.choice(
            [
                f"{labels[0]}: Hi, this is {{name}} from {{company}}. I'm calling to confirm your appointment on {{date}}. {labels[1]}: Yes, I have that scheduled. {labels[0]}: Great! We'll see you then. {labels[1]}: Thank you for confirming.",
                f"{labels[0]}: Hello, this is {{name}} from {{organization}}. We're conducting a customer satisfaction survey. {labels[1]}: How long will this take? {labels[0]}: Just a few minutes. Would you be willing to participate? {labels[1]}: Sure, I have time.",
                f"{labels[0]}: Good morning, this is {{name}} from {{service}}. I'm calling to remind you about your upcoming {{event}}. {labels[1]}: Oh, thank you for the reminder. {labels[0]}: You're welcome. Have a great day! {labels[1]}: You too.",
            ]
        )

        text = template.format(
            name=fake.first_name(),
            company=random.choice(["your doctor's office", "the pharmacy", "your insurance company"]),
            organization=random.choice(["your local clinic", "the community center"]),
            service=random.choice(["appointment reminder service", "delivery service"]),
            date=fake.date_this_month().strftime("%B %d"),
            event=random.choice(events),
        )

        return text, 0


def generate_synthetic_dataset(
    num_samples: int = 5000, fraud_ratio: float = 0.7, output_path: str = None
) -> pd.DataFrame:
    """
    Generate a synthetic dataset with varied conversation types

    Args:
        num_samples: Total number of samples
        fraud_ratio: Proportion of fraudulent calls
        output_path: Path to save CSV

    Returns:
        DataFrame with 'text' and 'label' columns
    """
    num_fraud = int(num_samples * fraud_ratio)
    num_legit = num_samples - num_fraud

    data = []

    logger.info(
        "Generating fraudulent conversations",
        total=num_fraud,
        one_way=int(num_fraud * 0.1),
        three_way=int(num_fraud * 0.1),
        edge_case=int(num_fraud * 0.8),
        templates=len(all_edge_case_templates),
    )

    for i in range(num_fraud):
        text, label = generate_fraud_call()
        data.append({"text": text, "label": label})
        if (i + 1) % 500 == 0:
            logger.info("Fraud generation progress", completed=i + 1, total=num_fraud)

    logger.info("Generating legitimate conversations", total=num_legit)
    for i in range(num_legit):
        text, label = generate_legit_call()
        data.append({"text": text, "label": label})
        if (i + 1) % 500 == 0:
            logger.info("Legit generation progress", completed=i + 1, total=num_legit)

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
        logger.info("Dataset saved", path=output_path)

    return df


def analyze_dataset(df: pd.DataFrame) -> None:
    """Analyze the generated dataset"""
    fraud_count = len(df[df["label"] == 1])
    legit_count = len(df[df["label"] == 0])
    total = len(df)

    logger.info(
        "Dataset statistics",
        total_samples=total,
        fraudulent=fraud_count,
        fraudulent_pct=f"{fraud_count/total*100:.1f}%",
        legitimate=legit_count,
        legitimate_pct=f"{legit_count/total*100:.1f}%",
    )

    # Analyze conversation types
    fraud_df = df[df["label"] == 1]
    legit_df = df[df["label"] == 0]

    oneway_fraud = fraud_df[
        fraud_df["text"].str.contains(
            "system:|voice:|bot:|voicemail:|automated_msg:|message:|automated_call:",
            case=False,
            na=False,
        )
    ]
    threeway_fraud = fraud_df[
        fraud_df["text"].str.contains("accomplice:|agent2:|manager:|supervisor:", case=False, na=False)
    ]

    oneway_legit = legit_df[
        legit_df["text"].str.contains(
            "system:|voice:|bot:|voicemail:|automated_msg:|message:",
            case=False,
            na=False,
        )
    ]
    threeway_legit = legit_df[legit_df["text"].str.contains("agent2:|coordinator:", case=False, na=False)]

    logger.info(
        "Conversation types (Fraud)",
        one_way=len(oneway_fraud),
        one_way_pct=f"{len(oneway_fraud)/len(fraud_df)*100:.1f}%",
        three_way=len(threeway_fraud),
        three_way_pct=f"{len(threeway_fraud)/len(fraud_df)*100:.1f}%",
        two_way=len(fraud_df) - len(oneway_fraud) - len(threeway_fraud),
    )

    logger.info(
        "Conversation types (Legitimate)",
        one_way=len(oneway_legit),
        one_way_pct=f"{len(oneway_legit)/len(legit_df)*100:.1f}%",
        three_way=len(threeway_legit),
        three_way_pct=f"{len(threeway_legit)/len(legit_df)*100:.1f}%",
        two_way=len(legit_df) - len(oneway_legit) - len(threeway_legit),
    )


# ============================================================================
# MAIN EXECUTION
# ============================================================================

from src.config import DATASETS_DIR

# ... (existing imports)

if __name__ == "__main__":
    logger.info(
        "Enhanced Synthetic Fraud Detection Dataset Generator",
        edge_case_templates=len(all_edge_case_templates),
        features=[
            "One-way conversations (voicemails, automated messages)",
            "Three-way conversations (scammer + accomplice + victim)",
            "Varied speaker labels (agent/customer, scammer/victim, etc.)",
            "US and Indian scam scenarios",
            "Code-switching (Hinglish) examples",
        ],
    )

    output_path = DATASETS_DIR / "fraudeye_synthetic.csv"
    df = generate_synthetic_dataset(num_samples=5000, fraud_ratio=0.7, output_path=output_path)

    analyze_dataset(df)

    # Generate dedicated edge case dataset for evaluation
    logger.info("Generating dedicated edge case dataset for evaluation")
    edge_case_data = []
    
    # Generate 500 Scam Edge Cases
    for _ in range(500):
        text, label = generate_edge_case()
        edge_case_data.append({"text": text, "label": label})
        
    # Generate 500 Legit Cases (to serve as negative examples)
    for _ in range(500):
        text, label = generate_legit_call()
        edge_case_data.append({"text": text, "label": label})
    
    # Shuffle the dataset
    random.shuffle(edge_case_data)
    
    edge_df = pd.DataFrame(edge_case_data)
    edge_output_path = DATASETS_DIR / "fraudeye_edge_case.csv"
    edge_df.to_csv(edge_output_path, index=False, quoting=csv.QUOTE_ALL)
    logger.info("Edge case dataset saved", path=str(edge_output_path), count=len(edge_df))

    logger.info("Generation complete")
