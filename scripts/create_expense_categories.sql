-- Run as postgres in pgAdmin on database expence (or expense_user after GRANT CREATE on public)

CREATE TABLE IF NOT EXISTS public.expense_categories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL UNIQUE,
    slug VARCHAR(100) NOT NULL UNIQUE,
    display_order INTEGER NOT NULL DEFAULT 0,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_expense_categories_slug ON public.expense_categories (slug);

INSERT INTO public.expense_categories (name, slug, display_order, is_active) VALUES
    ('Food and Dining', 'food-and-dining', 0, true),
    ('Groceries', 'groceries', 1, true),
    ('Rent', 'rent', 2, true),
    ('Utilities', 'utilities', 3, true),
    ('Fual', 'fual', 4, true),
    ('Shopping', 'shopping', 5, true),
    ('Entertainment', 'entertainment', 6, true),
    ('Healthcare', 'healthcare', 7, true),
    ('Edication', 'edication', 8, true),
    ('Personal care', 'personal-care', 9, true),
    ('Subscription', 'subscription', 10, true),
    ('EMI/Loans', 'emi-loans', 11, true),
    ('Insurance', 'insurance', 12, true),
    ('Investment', 'investment', 13, true),
    ('Travel', 'travel', 14, true),
    ('Savings', 'savings', 15, true),
    ('CAB/Taxi', 'cab-taxi', 16, true),
    ('Misecellaneous', 'misecellaneous', 17, true),
    ('Other', 'other', 18, true)
ON CONFLICT (name) DO NOTHING;

GRANT SELECT, INSERT, UPDATE, DELETE ON public.expense_categories TO expense_user;
