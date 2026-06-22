-- Run in pgAdmin Query Tool as postgres (superuser) on database: expence

\c expence

-- PG 15+: allow expense_user to create tables in public
GRANT USAGE ON SCHEMA public TO expense_user;
GRANT CREATE ON SCHEMA public TO expense_user;
ALTER SCHEMA public OWNER TO expense_user;

-- Move existing categories table from app_data → public
ALTER TABLE IF EXISTS app_data.expense_categories SET SCHEMA public;

-- Optional cleanup
DROP SCHEMA IF EXISTS app_data;

-- Verify
SELECT schemaname, tablename FROM pg_tables WHERE tablename = 'expense_categories';
SELECT COUNT(*) FROM public.expense_categories;
